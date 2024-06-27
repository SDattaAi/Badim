import pandas as pd
import clickhouse_driver
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
from matplotlib.lines import Line2D

###### inputs ######
start_date_for_check_results = '2023-06-01'
end_date_for_check_results = '2024-05-31'
unique_ids = ['20009902', '20005904', '20024900', 'a']
number_of_months_to_predict = 5
metric = 'MAE'
###############

password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)
month_to_predict = pd.to_datetime(end_date_for_check_results).strftime('%Y-%m')
month_to_predict = (pd.to_datetime(month_to_predict) + pd.DateOffset(months=number_of_months_to_predict)).strftime('%Y-%m')
end_month_to_predict = (pd.to_datetime(month_to_predict) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
df = client.query_dataframe(f'''SELECT * FROM platinum_badim.SWA_results
                            WHERE toDate(ds) >= %(start_date_for_check_results)s AND toDate(ds) <= %(end_month_to_predict)s 
                            AND unique_id in (%(unique_ids)s)''', {'unique_ids': unique_ids,
                                'start_date_for_check_results': start_date_for_check_results,
                                'end_month_to_predict': end_month_to_predict})
df['ds'] = pd.to_datetime(df['ds'])
colors = list(mcolors.TABLEAU_COLORS.keys())  # Get a list of colors from matplotlib's color palette
color_map = {unique_id: colors[i % len(colors)] for i, unique_id in enumerate(unique_ids)}  # Map unique_id to colors
line_style_legend = [Line2D([0], [0], color='black', linestyle='-', label='Actual'),
                     Line2D([0], [0], color='black', linestyle='--', label='Best Model')]
plt.figure(figsize=(15, 7))
final_ds_to_plot = pd.DataFrame()
for unique_id in unique_ids:
    if unique_id not in df['unique_id'].unique():
        print("unique_id", unique_id, "not in the data")
        continue
    df_unique_id_for_check = df[(pd.to_datetime(df['ds']) >= start_date_for_check_results) & (pd.to_datetime(df['ds']) <= end_month_to_predict) & (df['unique_id'] == unique_id)]
    df_unique_id_for_check_with_end_date = df_unique_id_for_check[df_unique_id_for_check['ds'] == end_month_to_predict]
    w_s_list_to_delete = df_unique_id_for_check_with_end_date[df_unique_id_for_check_with_end_date['SWA_value'].isnull()][['w', 's']].values
    for w_s in w_s_list_to_delete:
        df_unique_id_for_check = df_unique_id_for_check[(df_unique_id_for_check['w'] != w_s[0]) | (df_unique_id_for_check['s'] != w_s[1])]
    df_unique_id_for_check_g = df_unique_id_for_check.groupby(['w', 's']).agg({metric: 'mean'}).reset_index().rename(columns={metric: f'{metric}_mean'})
    best_w_s = df_unique_id_for_check_g[f'{metric}_mean'].idxmin()
    if pd.isnull(best_w_s):
        print("no best model for unique_id", unique_id)
        continue
    best_results = df_unique_id_for_check_g.iloc[best_w_s]
    best_w = int(best_results['w'])
    best_s = int(best_results['s'])
    best_result = df[(df['w'] == best_w) & (df['s'] == best_s) & (df['unique_id'] == unique_id) & (
            pd.to_datetime( df['ds']) >= start_date_for_check_results) & (pd.to_datetime(df['ds']) <= month_to_predict)].sort_values('ds')
    best_result['ds'] = best_result['ds'].astype(str).str[:7]
    color = color_map[unique_id]
    final_ds_to_plot = pd.concat([final_ds_to_plot, best_result])
final_ds_to_plot_swa = final_ds_to_plot.pivot_table(index='ds', columns='unique_id', values='SWA_value', aggfunc='first')
final_ds_to_plot_y = final_ds_to_plot.pivot_table(index='ds', columns='unique_id', values='y', aggfunc='first')
plt.figure(figsize=(15, 7))
colors = [color_map[col] for col in final_ds_to_plot_swa.columns]
final_ds_to_plot_y = final_ds_to_plot_y.reindex(final_ds_to_plot_swa.index)
print("final_ds_to_plot_y", final_ds_to_plot_y)
# Plotting
fig, ax = plt.subplots(figsize=(15, 7))
final_ds_to_plot_swa.plot(ax=ax, color=colors, linestyle='--')
final_ds_to_plot_y.plot(ax=ax, color=colors, linestyle='-')
color_legend_handles = [mlines.Line2D([0], [0], color=color_map[col], lw=2, label=col) for col in final_ds_to_plot_swa.columns]
style_legend_handles = [
    mlines.Line2D([0], [0], color='black', linestyle='--', lw=2, label='SWA Prediction'),
    mlines.Line2D([0], [0], color='black', linestyle='-', lw=2, label='Actual')
]

# Add legends to the plot
legend1 = ax.legend(handles=color_legend_handles, title='Unique IDs', loc='upper left')
ax.add_artist(legend1)
legend2 = ax.legend(handles=style_legend_handles, title='Line Style', loc='upper right')
plt.title('SWA values and y values, metric: ' + metric)
plt.show()