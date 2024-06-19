import os
import clickhouse_driver
import numpy as np
import pandas as pd
from sdtla.quey_utils import filter_for_query, filter_from_right_item_charachters
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
end_date = '2024-05-31'
agg_time_freq = 'M'
items = []
digits_0_2 = []
digits_2_5 = []
digits_2_8 = []
inv_mov_types = ['החזרה מלקוח', 'חשבוניות מס', 'דאטה מסאפ', 'משלוחים ללקוח']
start_date_test = '2024-01-01'



print(password, username, port, host)
client_name = 'badim'
layer = 'silver'
database = f'{layer}_{client_name}'
final_results = {}
max_number_of_seasons = 12
max_number_of_windows = 12
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
    results_linear_no_gradient = {}
    len_unique_ids = len(all_data['unique_id'].unique())
    number_of_folds_in_future = max_number_of_seasons
    max_date = all_data['ds'].max()
    max_date_plus_1_end_of_month = pd.to_datetime(max_date) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    range_of_end_dates = pd.date_range(start=max_date_plus_1_end_of_month, periods=number_of_folds_in_future,
                                        freq=agg_time_freq)
    # a
    print("range_of_end_dates", range_of_end_dates)
    i = 1
    results = []

    for unique_id in all_data['unique_id'].unique():

        print("unique_id:", unique_id)
        print(f"{i}/{len_unique_ids}")
        i += 1
        unique_id_data = all_data[all_data['unique_id'] == unique_id]
        # fill missing dates
        unique_id_data = unique_id_data.set_index('ds').resample(agg_time_freq).asfreq().reset_index()
        # add range_of_end_dates
        unique_id_data = pd.concat([unique_id_data, pd.DataFrame({'ds': range_of_end_dates})])
        unique_id_data['unique_id'] = unique_id
        unique_id_data = unique_id_data.sort_values('ds')
        unique_id_data['ds'] = unique_id_data['ds'].dt.strftime('%Y-%m-%d')
        for w in range(1, max_number_of_windows + 1):
            for s in range(1, max_number_of_seasons + 1):
                df_with_w_s = unique_id_data.copy()
                df_with_w_s['s'] = s
                df_with_w_s['w'] = w
                df_with_w_s['SWA_value'] = df_with_w_s['y'].shift(s).rolling(w).mean()

                results.append(df_with_w_s)



    df_all = pd.concat(results, ignore_index=True)

    df_all['MAE'] = np.abs(df_all['y'] - df_all['SWA_value'])
    df_all['MAPE'] = np.abs(df_all['MAE']) / df_all['y']
    df_all = df_all.sort_values(['unique_id', 'ds', 's', 'w'])
    # save df_all
    df_all = df_all[['ds', 'unique_id', 'y', 's', 'w', 'SWA_value', 'MAE', 'MAPE']]
    # upload to cklickhouse

    print("uploading to clickhouse...")
    client.execute('INSERT INTO  platinum_badim.SWA_results VALUES', df_all.to_dict(orient='records'))
    df_all.to_csv('SWA_results.csv')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # find for each unique_id the mean s and w combination in the last year [2023-06-01, 2024-05-31], after that give the best s and w combination for each unique_id
    start_date_for_check_results = '2023-06-01'
    df_for_check = df_all[(df_all['ds'] >= start_date_for_check_results) & (df_all['ds'] <= '2024-05-31')].groupby(['unique_id', 's', 'w'])[
        'MAE'].mean().reset_index().rename(columns={'MAE': 'MAE_mean'})
    best_w_s = df_for_check.groupby('unique_id')['MAE_mean'].idxmin()
    best_results = df_for_check.loc[best_w_s]
    best_w = best_results['w']
    best_s = best_results['s']


    print("df_for_check", df_for_check)
    print("best_results", best_results)
    print("best_w", best_w)
    print("best_s", best_s)





