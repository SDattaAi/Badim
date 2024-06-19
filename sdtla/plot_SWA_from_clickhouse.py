import pandas as pd
import clickhouse_driver
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

start_date_for_check_results = '2023-06-01'
end_date_for_check_results = '2024-05-31'
unique_ids = ['20_category']



password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)
unique_ids_format = ', '.join([f"'{unique_id}'" for unique_id in unique_ids])
df = client.query_dataframe('SELECT * FROM platinum_badim.SWA_results'
                            ' where unique_id in ({})'.format(unique_ids_format))
df['ds'] = pd.to_datetime(df['ds'])
print(df.info())
print(df)
df_for_check = df[(df['ds'] >= start_date_for_check_results) & (df['ds'] <= end_date_for_check_results)].groupby(['unique_id', 's', 'w'])[
    'MAE'].mean().reset_index().rename(columns={'MAE': 'MAE_mean'})
best_w_s = df_for_check.groupby('unique_id')['MAE_mean'].idxmin()
best_results = df_for_check.loc[best_w_s]
best_w = best_results['w'].iloc[0]
best_s = best_results['s'].iloc[0]


best_result = df[(df['w'] == best_w) & (df['s'] == best_s) & (df['unique_id'] == '20_category') & (df['ds'] >= start_date_for_check_results)].sort_values('ds')
# plot best result


# Assuming unique_ids, best_w, best_s, and start_date_for_check_results are already defined
colors = list(mcolors.TABLEAU_COLORS.keys())  # Get a list of colors from matplotlib's color palette
color_map = {unique_id: colors[i % len(colors)] for i, unique_id in enumerate(unique_ids)}  # Map unique_id to colors
from matplotlib.lines import Line2D



# Create a custom legend for line styles
line_style_legend = [Line2D([0], [0], color='black', linestyle='-', label='Actual'),
                     Line2D([0], [0], color='black', linestyle='--', label='Best Model')]

plt.figure()

for unique_id in unique_ids:
    best_result = df[
        (df['w'] == best_w) &
        (df['s'] == best_s) &
        (df['unique_id'] == unique_id) &
        (df['ds'] >= start_date_for_check_results)
        ].sort_values('ds')

    color = color_map[unique_id]  # Get the color for the current unique_id

    # Plot y with a regular line
    plt.plot(best_result['ds'], best_result['y'], label=f'Actual: {unique_id}', linestyle='-', color=color)

    # Plot SWA_value with a dashed line
    plt.plot(best_result['ds'], best_result['SWA_value'], label=f'Best Model for: {unique_id}', linestyle='--', color=color)

# Add the legend for unique IDs with different colors
unique_id_legend = [Line2D([0], [0], color=color_map[unique_id], lw=2, label=f'Unique_id: {unique_id}')
                    for unique_id in unique_ids]
unique_id_legend = plt.legend(handles=unique_id_legend, loc='upper right')
plt.gca().add_artist(unique_id_legend)

# Add the legend for the line styles
plt.legend(handles=line_style_legend, loc='upper left')

# Show the plot
plt.show()