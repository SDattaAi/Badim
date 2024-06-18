import os
import clickhouse_driver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasetsforecast.losses import mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sdtla.quey_utils import filter_for_query, filter_from_right_item_charachters
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, WindowAverage, SeasonalWindowAverage
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
items = []
digits_0_2 = []
digits_2_5 = []
digits_2_8 = []
inv_mov_types = ['החזרה מלקוח', 'חשבוניות מס', 'דאטה מסאפ', 'משלוחים ללקוח']
start_date_test = '2024-01-01'
num_of_folds = 36

models = []

# Add SeasonalNaive models with season_length ranging from 1 to 12
for number in range(1, 13):
    for number2 in range(1, 13):
        models.append(SeasonalWindowAverage(window_size=number, season_length=number2,
                                            alias=f'seasonal_window_average_{number}_{number2}'))

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

    # Train data preparation
    all_data = sales_df_grouped.copy()

    for unique_id in all_data['unique_id'].unique():
        print("unique_id:", unique_id)
        all_data_unique_id = all_data[all_data['unique_id'] == unique_id]
        train = all_data_unique_id[(all_data_unique_id['ds'] < start_date_test)]
        test = all_data_unique_id[(all_data_unique_id['ds'] >= start_date_test)]
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
        unique_id_crossvalidation_df = crossvalidation_df[crossvalidation_df.index == unique_id]
        cutoff_list = unique_id_crossvalidation_df['cutoff'].unique()
        all_mae_train = {}
        all_mape_train = {}
        for k in range(min(len(cutoff_list), num_of_folds)):
            cv = unique_id_crossvalidation_df[unique_id_crossvalidation_df['cutoff'] == cutoff_list[k]]
            cutoff = cutoff_list[k]
            cv = cv.drop(columns='cutoff')
            cv = cv.reset_index(drop=True)
            cv = cv.set_index('ds')
            # turn column to rows
            print(cv.loc[:, cv.columns != 'cutoff'].transpose())
            # replace nan to np.nan

            df_mae = cv.loc[:, cv.columns != 'cutoff']
            df_mape = cv.loc[:, cv.columns != 'cutoff']
            # replace nan to -1
            df_mae = df_mae.fillna(-1).apply(lambda x: mae(x, cv["y"]), axis=0)
            df_mape = df_mape.fillna(-1).apply(lambda x: mape(x, cv["y"]), axis=0)
            k_maes_train = df_mae.to_dict()
            k_mape_train = df_mape.to_dict()
            all_mae_train[k] = k_maes_train
            all_mape_train[k] = k_mape_train

        folds_mae_train = pd.DataFrame(all_mae_train)
        folds_mapes_train = pd.DataFrame(all_mape_train)
        # set column name to train
        crossvalidation_df_for_test = model.cross_validation(
            df=all_data_unique_id,
            h=1,
            step_size=1,
            n_windows=len(test)
        )
        all_mae_test = {}
        all_mape_test = {}
        for j in range(len(test) - 1):
            cv = crossvalidation_df_for_test[crossvalidation_df_for_test['cutoff'] == test['ds'].iloc[j]]
            cv = cv.drop(columns='cutoff')
            cv = cv.reset_index(drop=True)
            cv = cv.set_index('ds')
            df_mae = cv.loc[:, cv.columns != 'cutoff']
            df_mape = cv.loc[:, cv.columns != 'cutoff']
            df_mae = df_mae.fillna(-1).apply(lambda x: mae(x, cv["y"]), axis=0)
            df_mape = df_mape.fillna(-1).apply(lambda x: mape(x, cv["y"]), axis=0)
            k_maes_test = df_mae.to_dict()
            k_mapes_test = df_mape.to_dict()
            all_mae_test[j] = k_maes_test
            all_mape_test[j] = k_mapes_test

        folds_mae_test = pd.DataFrame(all_mae_test)
        folds_mapes_test = pd.DataFrame(all_mape_test)
        # add for columns + k
        folds_mae_test.columns = [int(col) + int(k + 1) for col in folds_mae_test.columns]
        folds_mapes_test.columns = [int(col) + int(k + 1) for col in folds_mapes_test.columns]
        # merge left
        folds_mae = folds_mae_train.merge(folds_mae_test, left_index=True, right_index=True, how='left')
        folds_mapes = folds_mapes_train.merge(folds_mapes_test, left_index=True, right_index=True, how='left')
        print("folds_mae.columns", folds_mae.columns)

        print("folds_mae", folds_mae)
        print("folds_mapes", folds_mapes)
        final_results[unique_id] = {}

        final_results[unique_id]['MAE'] = folds_mae.to_dict()
        final_results[unique_id]['MAPE'] = folds_mapes.to_dict()
        # plot heatmap for each unique_id
        # fig, ax = plt.subplots(figsize=(16, 10))
        # ax.imshow(folds_mae, cmap='viridis', interpolation='nearest')
        # ax.set_xticks(np.arange(folds_mae.shape[1]))
        # ax.set_yticks(np.arange(folds_mae.shape[0]))
        # ax.set_xticklabels(folds_mae.columns)
        # ax.set_yticklabels(folds_mae.index)
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # plt.show()

# save the results as pickle
import pickle

with open('no_gradient_models.pickle', 'wb') as f:
    pickle.dump(final_results, f)
