import pandas as pd
import clickhouse_driver
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


start_date_for_check_results = '2023-06-01'
end_date_for_check_results = '2024-05-31'
unique_ids = ['20009902', '20005904', '20024900', '20064111']
month_to_predict = '2024-06'



password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)
unique_ids_format = ', '.join([f"'{unique_id}'" for unique_id in unique_ids])
df = client.query_dataframe('SELECT * FROM platinum_badim.SWA_results'
                            ' where unique_id in ({})'.format(unique_ids_format))
df['ds'] = pd.to_datetime(df['ds'])
end_of_month_to_predict = pd.to_datetime(month_to_predict) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
end_of_month_to_predict = end_of_month_to_predict.strftime('%Y-%m-%d')
# Assuming unique_ids, best_w, best_s, and start_date_for_check_results are already defined
colors = list(mcolors.TABLEAU_COLORS.keys())  # Get a list of colors from matplotlib's color palette
color_map = {unique_id: colors[i % len(colors)] for i, unique_id in enumerate(unique_ids)}  # Map unique_id to colors
line_style_legend = [Line2D([0], [0], color='black', linestyle='-', label='Actual'),
                     Line2D([0], [0], color='black', linestyle='--', label='Best Model')]

plt.figure(figsize=(15, 7))

for unique_id in unique_ids:
    df_unique_id_for_check = df[(df['ds'] >= start_date_for_check_results) & (df['ds'] <= end_date_for_check_results) & (df['unique_id'] == unique_id)].groupby(
        ['s', 'w'])[
        'MAE'].mean().reset_index().rename(columns={'MAE': 'MAE_mean'})
    best_w_s = df_unique_id_for_check['MAE_mean'].idxmin()
    best_results = df_unique_id_for_check.iloc[best_w_s]
    best_w = int(best_results['w'])
    best_s = int(best_results['s'])
    best_result = df[(df['w'] == best_w) & (df['s'] == best_s) & (df['unique_id'] == unique_id) & (
                df['ds'] >= start_date_for_check_results) & (df['ds'] <= end_of_month_to_predict)].sort_values('ds')
    color = color_map[unique_id]
    plt.plot(best_result['ds'], best_result['y'], label=f'Actual: {unique_id}', linestyle='-', color=color)
    plt.plot(best_result['ds'], best_result['SWA_value'], label=f'Best Model for: {unique_id}', linestyle='--', color=color)
    print(f"prediction of best model in {month_to_predict} of unique_id", unique_id, "is", np.round(df[(df['ds'].astype(str).str[:7] == month_to_predict) & (df['unique_id'] == unique_id)]['SWA_value'].values[0], 2), 'best_w:', best_w, 'best_s:', best_s)


# Add the legend for unique IDs with different colors
unique_id_legend = [Line2D([0], [0], color=color_map[unique_id], lw=2, label=f'Unique_id: {unique_id}')
                    for unique_id in unique_ids]
unique_id_legend = plt.legend(handles=unique_id_legend, loc='center left', bbox_to_anchor=(1, 0.5))
plt.gca().add_artist(unique_id_legend)

# Add the legend for the line styles
plt.legend(handles=line_style_legend, loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Actual vs Best Model')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.tight_layout()
# save the plot
plt.savefig('SWA_results_example.png')
# Show the plot
plt.show()
