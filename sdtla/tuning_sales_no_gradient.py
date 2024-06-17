import os
import clickhouse_driver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasetsforecast.losses import rmse
from sdtla.quey_utils import filter_for_query, filter_from_right_item_charachters
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, WindowAverage, SeasonalWindowAverage
import warnings


# Fetch environment variables
password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

# Parameters
start_date = '2019-01-01'
end_date = '2024-06-30'
agg_time_freq = 'M'
items = ['20009902', '20004028']
digits_0_2 = []
digits_2_5 = []
digits_2_8 = []
inv_mov_types = ['החזרה מלקוח', 'חשבוניות מס', 'דאטה מסאפ', 'משלוחים ללקוח']
start_date_test = '2024-01-01'
num_of_folds=36

models  = [
    Naive(alias='naive'),
]

# Add SeasonalNaive models with season_length ranging from 1 to 12
for number in range(1, 13):
    models.append(SeasonalNaive(season_length=number, alias=f'seasonal_naive_{number}'))
    models.append(WindowAverage(window_size=number, alias=f'window_average_{number}'))
    for number2 in range(1, 13):
        models.append(SeasonalWindowAverage(window_size=number, season_length=number2, alias=f'seasonal_window_average_{number}_{number2}'))


print(password, username, port, host)
client_name = 'badim'
layer = 'silver'
database = f'{layer}_{client_name}'

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
    print(query)
    sales_df = client.query_dataframe(query)
    # i want add unique id ass hierarchy of the first 2 digits

    print(sales_df.head())
    print(sales_df.columns)

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

    print("sales_df_grouped", sales_df_grouped)

    # Train data preparation
    all_data = sales_df_grouped.copy()

    for unique_id in all_data['unique_id'].unique():
        print("unique_id", unique_id)
        train = all_data[(all_data['unique_id'] == unique_id) & (all_data['ds'] < start_date_test)]
        test = all_data[(all_data['unique_id'] == unique_id) & (all_data['ds'] >= start_date_test)]
        model = StatsForecast(
            models=models,
            freq=agg_time_freq,
            n_jobs=-1
        )
        crossvalidation_df = model.cross_validation(
            df=train,
            h=1,
            step_size=1,
            n_windows=num_of_folds
        )

        # Fit the model
        for unique_id in train['unique_id'].unique():
            print("unique_id", unique_id)
            unique_id_crossvalidation_df = crossvalidation_df[crossvalidation_df.index == unique_id]
            cutoff_list = unique_id_crossvalidation_df['cutoff'].unique()
            all_rmse = {}
            for k in range(min(len(cutoff_list), num_of_folds)):  # Limit to 3 cutoffs
                cv = unique_id_crossvalidation_df[unique_id_crossvalidation_df['cutoff'] == cutoff_list[k]]
                cutoff = cutoff_list[k]
                cv = cv.drop(columns='cutoff')
                cv = cv.reset_index(drop=True)
                cv = cv.set_index('ds')
                train_for_plot = train[(train['unique_id'] == unique_id) & (train['ds'] <= cutoff)]
                k_rmses = cv.loc[:, cv.columns != 'cutoff'].apply(lambda x: rmse(x, cv["y"]), axis=0).to_dict()
                all_rmse[k] = k_rmses

            mean_rmse = pd.DataFrame(all_rmse).mean(axis=1)
            print("mean_rmse", mean_rmse.sort_values().dropna())

