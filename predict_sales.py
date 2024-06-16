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
start_date = '2024-04-01'
end_date = '2024-06-30'
agg_time_freq = 'D'

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

# Connect to ClickHouse
client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)

# Fetch data
sales_df = client.query_dataframe(f'SELECT * FROM {database}.sales')
print(sales_df.columns)
print(sales_df['order_status'].unique())

# Filter and prepare data
sales_df = sales_df[sales_df['order_status'].isin(['שולמה', 'בוצעה'])].copy()
sales_df['status_date'] = pd.to_datetime(sales_df['status_date'])
sales_df['total_price'] = sales_df['total_price'].astype(float)
sales_df_grouped = sales_df.groupby(['status_date', 'item'])['total_price'].sum().unstack().fillna(0).stack().reset_index()
sales_df_grouped = sales_df_grouped.rename(columns={'status_date': 'ds', 'item': 'unique_id', 0: 'y'})

print(sales_df_grouped)

# Train data preparation
train = sales_df_grouped.copy()
train = train[train['unique_id'].isin(['10000000'])]
print("train", train)

# Initialize the StatsForecast model with the desired models
model = StatsForecast(
    models=[
        Naive(),
        SeasonalNaive(season_length=7),
        WindowAverage(window_size=7),
        SeasonalWindowAverage(window_size=2, season_length=7),
    #    CustomLagModel(lags=7)
    ],
    freq=agg_time_freq,
    n_jobs=-1
)

# Fit the model
model.fit(train)

# Generate the forecast
forecast = model.predict(h=30)

# Add lag_7 column to forecast
print("forecast", forecast)
forecast = forecast.reset_index()
# Plot forecast and train data
for unique_id in train['unique_id'].unique():
    print("unique_id", unique_id)
    unique_id_train = train[train['unique_id'] == unique_id]
    unique_id_forecast = forecast.loc[forecast['unique_id'] == unique_id]
    plt.figure(figsize=(10, 6))
    plt.plot(unique_id_train['ds'], unique_id_train['y'], label='Train')
    plt.plot(unique_id_forecast['ds'], unique_id_forecast['Naive'], label='Naive')
    plt.plot(unique_id_forecast['ds'], unique_id_forecast['SeasonalNaive'], label='SeasonalNaive')
    plt.plot(unique_id_forecast['ds'], unique_id_forecast['WindowAverage'], label='WindowAverage')
    plt.plot(unique_id_forecast['ds'], unique_id_forecast['SeasWA'], label='SeasWA')
    plt.title(f'Unique ID: {unique_id}')
    plt.legend()
    plt.show()
