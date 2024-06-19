import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


with open('results_linear_no_gradient.pkl', 'rb') as f:
    results_linear_no_gradient = pickle.load(f)
# replace -1 with nan
last_date = '2024-06-01'
#
# filter keys that end with 'category'
keys_with_category_from_end = [key for key in results_linear_no_gradient.keys() if key.endswith('category')]
len_keys = len(results_linear_no_gradient.keys())
i = 1
all_data_df = pd.DataFrame()
for unique_id in ['20_category']:
    print(unique_id)
    print(f'{i}/{len_keys}')
    i += 1
    unique_id_data = pd.DataFrame(results_linear_no_gradient[unique_id]['unique_id_data'])
    # mae_for_each_fold = pd.DataFrame(results_linear_no_gradient[unique_id]['mae_for_each_fold'])
    # # mape_for_each_fold = pd.DataFrame(results_linear_no_gradient[unique_id]['mape_for_each_fold'])
    # # best_models_sortbest_models_sort is the best model sorted by mae mean
    #
    # best_models_sort =  mae_for_each_fold.mean().sort_values()
    # sns.heatmap(mae_for_each_fold[best_models_sort.index].T)
    # plt.title(f'MAE for each fold for unique_id: {unique_id}')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # log_mape_data = np.log10(mape_for_each_fold[best_models_sort.index].T)
    #
    # # Plot the heatmap
    # sns.heatmap(log_mape_data)
    # plt.title(f'MAPE for each fold for unique_id: {unique_id} (log10)')
    # plt.tight_layout()
    #
    # plt.show()
    #
    #
    # # plot the best model
    # best_model = best_models_sort.index[0]
    # plt.plot(unique_id_data['ds'], unique_id_data['y'], label='y')
    # plt.plot(unique_id_data['ds'], unique_id_data[best_model], label='Best model')
    # plt.legend()
    # plt.title(f'Best model for unique_id: {unique_id}')
    # plt.tight_layout()
    # plt.show()
    unique_id_data['ds'] = unique_id_data['ds'].dt.strftime('%Y-%m-%d')
    unique_id_data['data_type'] = 'true_and_pred'
    unique_id_data['unique_id'] = unique_id

    all_data_df = pd.concat([all_data_df, unique_id_data])
print("all_data_df.head()", all_data_df.head())
df_melted = all_data_df.melt(id_vars=['unique_id', 'ds', 'y', 'data_type'], var_name='variable', value_name='value')
print("df_melted1", df_melted.head())
# Extracting 's' and 'w' values from 'variable' column
df_melted[['prefix', 's', 'w']] = df_melted['variable'].str.extract(r'(SWA)_s(\d+)_w(\d+)')
print("df_melted2", df_melted.head())
# Dropping the 'variable' and 'prefix' columns as they are no longer needed
df_melted = df_melted.drop(columns=['variable', 'prefix'])
# Step 1: Print unique values in 'data_type' from melted DataFrame
print("Unique values in 'data_type':", df_melted['data_type'].unique())
print("df_melted3", df_melted.head()    )

#replace NaN with -1

df_melted['y'] = df_melted['y'].fillna(-1)
#show nan values
print("df_melted.isnull().sum()", df_melted.isnull().sum())
print("df_melted4", df_melted.tail())
# Step 2: Pivot the DataFrame and print columns
df_pivot = df_melted.pivot_table(
    index=['unique_id', 'ds', 'y', 's', 'w'],
    columns='data_type',
    values='value',
    aggfunc='first',
    dropna=False
).reset_index()

print("df_pivot['ds'].sort_values().unique()", df_pivot['ds'].sort_values().unique())
print("df_melted['ds'].sort_values().unique()", df_melted['ds'].sort_values().unique())



print("Columns after pivoting:", df_pivot.columns)

# Renaming the columns for clarity
df_pivot = df_pivot.rename(columns={'true_and_pred': 'SWA_value'})

# Reordering the columns as per requirement
df_final = df_pivot[['unique_id', 'ds', 'y', 's', 'w', 'SWA_value']]
df_final['y'] = df_final['y'].replace(-1, np.nan)
df_final['MAE'] = np.abs(df_final['SWA_value'] - df_final['y'])
df_final['MAPE'] = df_final['MAE'] / df_final['y']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df_final[(df_final['unique_id'] == '20_category') & (df_final['s'] == '9') & (df_final['w'] == '9')])

# save the df_final
df_final.to_csv('no_gradient_results.csv', index=False)