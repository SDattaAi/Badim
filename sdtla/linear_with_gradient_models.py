import os
import clickhouse_driver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasetsforecast.losses import mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sdtla.quey_utils import filter_for_query, filter_from_right_item_charachters
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import warnings

# drop warinings
warnings.filterwarnings('ignore')

# Fetch environment variables
password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

# Parameters
start_date = '2019-01-01'
end_date = '2024-06-30'
agg_time_freq = 'M'
items = ['20009902']
digits_0_2 = []
digits_2_5 = []
digits_2_8 = []
inv_mov_types = ['החזרה מלקוח', 'חשבוניות מס', 'דאטה מסאפ', 'משלוחים ללקוח']
start_date_test = '2024-01-01'
num_of_folds = 36

models = [AutoARIMA()]

print(password, username, port, host)
client_name = 'badim'
layer = 'silver'
database = f'{layer}_{client_name}'
final_results = {}
if __name__ == "__main__":
    # Connect to ClickHouse
    client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)

    query = f'''SELECT * FROM {database}.stock_log
                                      where toDate(cur_date) >= '{start_date}' and toDate(cur_date) <= '{end_date}'
                                        {filter_for_query('inv_mov_type', inv_mov_types)}
                                        {filter_for_query('item', items)}
                                                                    {filter_from_right_item_charachters(digits_0_2, 0, 2)}
                             {filter_from_right_item_charachters(digits_2_5, 2, 3)}
                             {filter_from_right_item_charachters(digits_2_8, 2, 6)}
                                      '''

    sales_df = client.query_dataframe(query)
    # i want add unique id ass hierarchy of the first 2 digits

    df_final = pd.DataFrame(columns=['ds', 'unique_id', 'y'])
    # Filter and prepare data
    sales_df['cur_date'] = pd.to_datetime(sales_df['cur_date'])
    sales_df['quantity'] = sales_df['quantity'].astype(float)
    sales_df_grouped = sales_df.groupby(['cur_date', 'item'])['quantity'].sum().unstack().fillna(
        0).stack().reset_index()
    # group by agg_time_freq
    sales_df_grouped = sales_df_grouped.groupby(['item', pd.Grouper(key='cur_date', freq=agg_time_freq)])[
        0].sum().reset_index()
    sales_df_grouped['ch_0_2'] = sales_df_grouped['item'].str[0:2] + '_category'
    sales_df_grouped['ch_2_5'] = sales_df_grouped['item'].str[2:5] + '_catalog'
    sales_df_grouped['ch_2_8'] = sales_df_grouped['item'].str[2:8] + '_catalog+color'

    # sales_df_grouped = sales_df_grouped.rename(columns={'cur_date': 'ds', 'item': 'unique_id', 0: 'y'})
    sales_df_grouped1 = sales_df_grouped.rename(columns={'cur_date': 'ds', 'item': 'unique_id', 0: 'y'})[
        ['ds', 'unique_id', 'y']]
    sales_df_grouped2 = sales_df_grouped.rename(columns={'cur_date': 'ds', 'ch_0_2': 'unique_id', 0: 'y'})[
        ['ds', 'unique_id', 'y']]
    sales_df_grouped3 = sales_df_grouped.rename(columns={'cur_date': 'ds', 'ch_2_5': 'unique_id', 0: 'y'})[
        ['ds', 'unique_id', 'y']]
    sales_df_grouped4 = sales_df_grouped.rename(columns={'cur_date': 'ds', 'ch_2_8': 'unique_id', 0: 'y'})[
        ['ds', 'unique_id', 'y']]
    sales_df_grouped = pd.concat([sales_df_grouped1, sales_df_grouped2, sales_df_grouped3, sales_df_grouped4],
                                 axis=0).groupby(['ds', 'unique_id']).sum().reset_index()

    sales_df_grouped = sales_df_grouped.sort_values(['unique_id', 'ds'])
    map_from_date_to_int = {date: i for i, date in enumerate(sales_df_grouped['ds'].sort_values().astype(str).unique())}
    print("map_from_date_to_int:", map_from_date_to_int)
    sales_df_grouped['ds'] = sales_df_grouped['ds'].astype(str).map(map_from_date_to_int)
    # find the last day in start_date_test month as yyyy-mm-dd
    last_day_start_date_test = pd.to_datetime(start_date_test) + pd.offsets.MonthEnd(0)
    last_day_start_date_test = last_day_start_date_test.strftime('%Y-%m-%d')
    start_date_test_int = map_from_date_to_int[last_day_start_date_test]
    print(sales_df_grouped)
    # Train data preparation
    all_data = sales_df_grouped.copy()
    train = all_data[all_data['ds'] < start_date_test_int]
    test = all_data[all_data['ds'] >= start_date_test_int]
    print("train:", train)
    print("test:", test)
    model = StatsForecast(
        models=models, freq=1, verbose=True)
    print(train['y'].tolist())
    model.fit(train)

    forecast = model.predict(h=len(test['ds'].unique())).reset_index()
    print(f'forecast: {forecast}')

    for unique_id in all_data['unique_id'].unique()[:100]:
        # plot train and test and prediction
        all_data_unique_id = all_data[all_data['unique_id'] == unique_id]
        train_unique_id = all_data_unique_id[(all_data_unique_id['ds'] < start_date_test_int)]
        test_unique_id = all_data_unique_id[(all_data_unique_id['ds'] >= start_date_test_int)]
        forecast_unique_id = forecast[forecast['unique_id'] == unique_id]
        plt.plot(train_unique_id['ds'], train_unique_id['y'], label='train')
        plt.plot(test_unique_id['ds'], test_unique_id['y'], label='test')
        plt.plot(forecast_unique_id['ds'], forecast_unique_id['AutoARIMA'], label='AutoARIMA')
        plt.legend()
        plt.show()

