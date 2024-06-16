import os
import clickhouse_driver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, WindowAverage, SeasonalWindowAverage

# Fetch environment variables
password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

# Parameters
start_date = '2023-05-01'
end_date = '2024-06-30'
agg_time_freq = 'M'
items = ['20009902', '20004028']
inv_mov_types = ['החזרה מלקוח','חשבוניות מס', 'דאטה מסאפ','משלוחים ללקוח']
def filter_for_query(name_of_column, filter_list):
    if len(filter_list) == 0:
        return ''
    else:
        # add quotes to string values
        formatted_values = ", ".join([f"'{x}'" if isinstance(x, str) else str(x) for x in filter_list])
        return f"AND {name_of_column} IN ({formatted_values})"

print(password, username, port, host)
client_name = 'badim'
layer = 'silver'
database = f'{layer}_{client_name}'

# Custom Lag model implementation
class CustomLagModel:
    def __init__(self, lags):
        self.lags = lags
        self.uses_exog = False  # Indicates that this model does not use exogenous variables
        self.__name__ = "CustomLagModel"

    def fit(self, y, X=None):
        self.y = y

    def predict(self, h, X=None):
        forecasts = []
        y_extended = self.y.copy()
        for i in range(h):
            forecast = y_extended.iloc[-self.lags]
            forecasts.append(forecast)
            y_extended = y_extended.append(pd.Series([forecast]), ignore_index=True)
        return pd.Series(forecasts)

    def new(self):
        return CustomLagModel(lags=self.lags)
if __name__ == "__main__":

    # Connect to ClickHouse
    client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)

    query = f'''SELECT * FROM {database}.stock_log
                                      where toDate(cur_date) >= '{start_date}' and toDate(cur_date) <= '{end_date}'
                                        {filter_for_query('inv_mov_type', inv_mov_types)}
                                        {filter_for_query('item', items)}
                                      '''
    print(query)
    sales_df = client.query_dataframe(query)

    print(sales_df.head())
    print(sales_df.columns)


    # Filter and prepare data
    sales_df['cur_date'] = pd.to_datetime(sales_df['cur_date'])
    sales_df['quantity'] = sales_df['quantity'].astype(float)
    sales_df_grouped = sales_df.groupby(['cur_date', 'item'])['quantity'].sum().unstack().fillna(0).stack().reset_index()
    # group by agg_time_freq
    sales_df_grouped = sales_df_grouped.groupby(['item', pd.Grouper(key='cur_date', freq=agg_time_freq)])[0].sum().reset_index()
    sales_df_grouped = sales_df_grouped.rename(columns={'cur_date': 'ds', 'item': 'unique_id', 0: 'y'})

    print("sales_df_grouped", sales_df_grouped)

    # Train data preparation
    train = sales_df_grouped.copy()


    # Initialize the StatsForecast model with the desired models
    model = StatsForecast(
        models=[
            Naive(),
            SeasonalNaive(season_length=12),
            WindowAverage(window_size=6),
            SeasonalWindowAverage(window_size=2, season_length=6),
        ],
        freq=agg_time_freq,
        n_jobs=-1
    )

    # Fit the model
    model.fit(train)

    # Generate the forecast
    forecast = model.predict(h=4)

    # Add lag_7 column to forecast
    forecast = forecast.reset_index()
    all_data = pd.concat([train, forecast], axis=0)
    print("all_data", all_data)
    # Plot forecast and train data
    for unique_id in train['unique_id'].unique():
        print("unique_id", unique_id)
        unique_id_all_data = all_data[all_data['unique_id'] == unique_id]
        plt.figure(figsize=(10, 6))
        plt.plot(unique_id_all_data['ds'], unique_id_all_data['y'], label='Train')
        plt.plot(unique_id_all_data['ds'], unique_id_all_data['Naive'], label='Naive')
        plt.plot(unique_id_all_data['ds'], unique_id_all_data['SeasonalNaive'], label='SeasonalNaive')
        plt.plot(unique_id_all_data['ds'], unique_id_all_data['WindowAverage'], label='WindowAverage')
        plt.plot(unique_id_all_data['ds'], unique_id_all_data['SeasWA'], label='SeasWA')
        plt.title(f'Unique ID: {unique_id}, agg_time_freq: {agg_time_freq}')
        plt.legend()
        plt.show()
