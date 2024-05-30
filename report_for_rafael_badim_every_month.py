import os
import clickhouse_connect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']
######
start_date = '2024-04-01'
end_date = '2024-05-30'
agg_time_freq = 'no_agg'
########
print(password, username, port, host)
client = 'badim'
layer = 'silver'
# Connection details
database = f'{layer}_{client}'
agg_date_update = agg_time_freq
if agg_time_freq == 'no_agg':
    agg_date_update = 'D'
date_trunc_func = {
    'D': 'toStartOfDay',
    'W': 'toStartOfWeek',
    'M': 'toStartOfMonth',
    'Q': 'toStartOfQuarter',
    'Y': 'toStartOfYear'
}.get(agg_date_update, 'toStartOfDay')

# Directory containing the SQL scripts
input_directory = f'/Users/guybasson/PycharmProjects/clickhouse_sql_repo/setup/{client}/generation_tables/{layer}'

# Connect to ClickHouse
client = clickhouse_connect.get_client(host=host, user=username, password=password, port=port, database=database)


sales_df = client.query_df(f'''SELECT * FROM silver_badim.sales
                            WHERE date >= '{start_date}' AND date <= '{end_date}'  ''')
# in sales_df i have date column that contains date in format '2021-01-01' or '2021-01' i want add column
# that will contain only year and month in format '2021-01'
sales_df['year_month'] = sales_df['date'].str[:7]
sales_df['year_month'] = pd.to_datetime(sales_df['year_month'])
sales_df['date'] = pd.to_datetime(sales_df['date'])

sales_df['sales'] = sales_df['sales'].astype(float)
sales_df['total_price'] = sales_df['total_price'].astype(float)
print(sales_df)
# show columns in sales_df
print(sales_df.columns)
# there is nan in sales column,show it
print("sales_df[sales_df['sales'].isnull()] ", sales_df[sales_df['sales'].isnull()])
print("sales_df[sales_df['total_price'].isnull()]", sales_df[sales_df['total_price'].isnull()])
print("Unique items with null total_price:", sales_df[sales_df['total_price'].isnull()]['item'].unique())
sales_df['total_price'] = sales_df['total_price'].astype(float)
sales_df['sales'] = sales_df['sales'].astype(float)
## plot 1 - total income per agg_time_freq - time series
plt.figure(figsize=(22, 10))  # You can adjust the dimensions as needed reindex to show all days and fill nan with 0
date_range = pd.date_range(start=sales_df.index.min(), end=sales_df.index.max())
# Reindex the DataFrame to include all dates in the range

print("1111121234234", sales_df.set_index('date')['total_price'].resample(agg_date_update).sum().fillna(0))
# Plot based on aggregation frequency
if agg_time_freq == 'D' or agg_time_freq == 'no_agg':
    sales_df.set_index('date')['total_price'].resample('D').sum().fillna(0).plot(kind='line')
else:
    sales_df.set_index('date')['total_price'].resample(agg_time_freq).sum().fillna(0).plot(kind='line')
# make plot 1 data with clickhouse query




################################################# Plot 1 - clickhouse data #################################################
query_1 = f'''
    SELECT {date_trunc_func}(toDate(date)) as agg_date, sum(total_price) as total_price
    FROM silver_badim.sales
    WHERE toDate(date) >= toDate('{start_date}') AND toDate(date) <= toDate('{end_date}')
    GROUP BY agg_date
    ORDER BY agg_date
'''
plot_1_data_clickhouse = client.query_df(query_1).set_index('agg_date')
# fill missing dates with 0
plot_1_data_clickhouse = plot_1_data_clickhouse.reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()

print("plot 1 data clickhouse", plot_1_data_clickhouse)



target_date = pd.to_datetime('2024-04-01')
plt.axvline(x=target_date, color='r', linestyle='--', label='from sap to priority')
plt.xticks(rotation=45)
plt.title('Plot 1 - Total income per agg_time_freq')
plt.legend()
# save plot 1 to png
plt.savefig('plot1.png')
plt.show()

plt.plot(plot_1_data_clickhouse)
plt.title('Plot 1.1 - Total income per agg_time_freq clickhouse')
plt.show()

# savle plot 1 to png
# plot 1.1 - clickhouse data
# plot 1.2 - clickhouse data

##############################################################################################################################
sales_df['date'] = pd.to_datetime(sales_df['date'])
# plot 2 - sales per day
plt.figure(figsize=(22, 10))  # You can adjust the dimensions as needed
if agg_time_freq == 'D' or agg_time_freq == 'no_agg':
    sales_df.set_index('date')['sales'].resample('D').sum().fillna(0).plot(kind='line')
else:
    sales_df.set_index('date')['sales'].resample(agg_time_freq).sum().fillna(0).plot(kind='line')
print("plot 2 data", sales_df.set_index('date')['sales'].resample(agg_date_update).sum().fillna(0))
plt.xticks(rotation=45)
plt.title('Plot 2 - Sales per agg_time_freq')
plt.legend()
# save plot 2 to png
plt.savefig('plot2.png')
plt.show()
################################################# Plot 2 - clickhouse data #################################################
query_2 = f'''
    SELECT {date_trunc_func}(toDate(date)) as agg_date, sum(sales) as total_sales
    FROM silver_badim.sales
    WHERE toDate(date) >= toDate('{start_date}') AND toDate(date) <= toDate('{end_date}')
    GROUP BY agg_date
    ORDER BY agg_date
'''
plot_2_data_clickhouse = client.query_df(query_2).set_index('agg_date')

# fill missing dates with 0
plot_2_data_clickhouse = plot_2_data_clickhouse.reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
# plot 2 - clickhouse data
plt.plot(plot_2_data_clickhouse)
plt.title('Plot 2.1 - Sales per agg_time_freq clickhouse')
plt.show()
##############################################################################################################################

sales_df['date'] = pd.to_datetime(sales_df['date'])
top_items = sales_df.groupby('item')['total_price'].sum().nlargest(7).index
sales_df_top_10_items = sales_df[sales_df['item'].isin(top_items)]
#
pivot_table = sales_df_top_10_items.pivot_table(values='total_price', index='date', columns='item', aggfunc='sum')
#
# add dates for missing dates and fill nan with 0 from start_date to end_date for each item
pivot_table = pivot_table.reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
print("plot 3 data", pivot_table)
# Plot 3 - Top 10 items with the highest sales
plt.figure(figsize=(13, 10))  # Adjust size as needed
pivot_table.plot(kind='line', ax=plt.gca())  # Plot all top items with different colors
plt.xticks(rotation=45)
plt.legend(title='Item', loc='upper left')  # Customize legend location and title as needed
plt.title('Plot 3 - Top 10 items with the highest income')  # Add a title to the plot
plt.axvline(x=target_date, color='r', linestyle='--', label='from sap to priority')
# save plot 3 to png
plt.savefig('plot3.png')
plt.show()
################################################# Plot 3 - clickhouse data #################################################
query_3 = f'''WITH
    '{start_date}' AS start_date,
    '{end_date}' AS end_date,
    -- Get top items
    top_items AS (
        SELECT item
        FROM silver_badim.sales
        WHERE date >= start_date AND date <= end_date
        GROUP BY item
        ORDER BY sum(total_price) DESC
        LIMIT 7
    ),
    -- Create a date range
    date_range AS (
        SELECT arrayJoin(arrayMap(x -> toDate(x), range(toUInt32(toDate(start_date)), toUInt32(toDate(end_date)) + 1))) AS date
    ),
    -- Cross join dates with top items
    cross_join AS (
        SELECT
            date,
            item
        FROM
            date_range
        CROSS JOIN
            top_items
    )
-- Generate the pivot table
SELECT
    cross_join.date,
    cross_join.item,
    sum(sales.total_price) AS total_price
FROM
    cross_join
LEFT JOIN
    (SELECT date, item, total_price FROM silver_badim.sales WHERE date >= start_date AND date <= end_date) AS sales
ON
    cross_join.date = toDate(sales.date) AND cross_join.item = sales.item
GROUP BY
    cross_join.date,
    cross_join.item
ORDER BY
    cross_join.date, cross_join.item

'''
plot_3_data_clickhouse = client.query_df(query_3)
# Pivot the data to get total price per day per item
plot_3_data_clickhouse = plot_3_data_clickhouse.pivot_table(values='total_price', index='date', columns='item', aggfunc='sum')
# add dates for missing dates and fill nan with 0 from start_date to end_date for each item
# resample to agg_time_freq
plot_3_data_clickhouse = plot_3_data_clickhouse.reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
plot_3_data_clickhouse = plot_3_data_clickhouse.astype(float)
print("plot 3 data clickhouse", plot_3_data_clickhouse)
# plot 3 - clickhouse data
plt.figure(figsize=(13, 10))  # Adjust size as needed
plot_3_data_clickhouse.plot(kind='line', ax=plt.gca())  # Plot all top items with different colors
plt.xticks(rotation=45)
plt.legend(title='Item', loc='upper left')  # Customize legend location and title as needed
plt.title('Plot 3.1 - Top 10 items with the highest income clickhouse')  # Add a title to the plot
plt.show()
##################################################################################################



# Plot 4 - Top 30 items with the highest sales and others pie chart
top_items = sales_df.groupby('item')['total_price'].sum().nlargest(10).index
sales_df_top_10_items = sales_df[sales_df['item'].isin(top_items)]
sales_df_top_10_items = sales_df_top_10_items.groupby('item').agg({'total_price': 'sum'})
sales_df_top_10_items.loc['others'] = sales_df[~sales_df['item'].isin(top_items)]['total_price'].sum()
sales_df_top_10_items.plot(kind='pie', y='total_price',labels=None, autopct='%1.1f%%', pctdistance=1.15)
# print data
print("plot 4 data", sales_df_top_10_items.sort_values('total_price', ascending=False))
plt.title('Plot 4 - Top 10 items with the highest sales and others')
# save plot 4 to png
plt.savefig('plot4.png')
plt.show()

########################################## Plot 4 - clickhouse data ########################################################
query_4 = f'''WITH
    '{start_date}' AS start_date,
    '{end_date}' AS end_date,
    -- Get top 10 items by total price
    top_items AS (
        SELECT item
        FROM silver_badim.sales
        WHERE date >= start_date AND date <= end_date
        GROUP BY item
        ORDER BY sum(total_price) DESC
        LIMIT 10
    )
-- Aggregate total price for top 10 items and calculate "others"
SELECT 
    multiIf(item IN (SELECT item FROM top_items), item, 'others') AS item,
    sum(total_price) AS total_price
FROM 
    silver_badim.sales
WHERE 
    date >= start_date AND date <= end_date
GROUP BY 
    item
ORDER BY
    total_price DESC'''

plot_4_data_clickhouse = client.query_df(query_4)
print("plot 4 data clickhouse", plot_4_data_clickhouse.sort_values('total_price', ascending=False))

# Plot the pie chart
plt.figure(figsize=(10, 7))
plt.pie(plot_4_data_clickhouse['total_price'], labels=plot_4_data_clickhouse['item'], autopct='%1.1f%%')
plt.title('Plot 4.1 - Top 10 items with the highest sales and others clickhouse')
# save plot 4 to png
plt.savefig('plot4.png')
plt.show()


##############################################################################################################################

# Plot 5 - Total sales per day of the week
# agg sum of total_price per day first do sum per date and then do mean per day of the week
sales_df_agg_d_s = sales_df.groupby('date').agg({'total_price': 'sum'}).reset_index()
sales_df_agg_d_s['day_of_week'] = sales_df_agg_d_s['date'].dt.dayofweek
sales_df_agg_d_s['day_of_week'] = sales_df_agg_d_s['day_of_week'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                                                                        3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                                                                        6: 'Sunday'})
# i want reindex sunday - sat to monday - sunday
sales_df_agg_w_d = sales_df_agg_d_s.groupby('day_of_week').agg({'total_price': 'mean'}).reindex(
    ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']).fillna(0)
print("plot 5 data", sales_df_agg_w_d)
sales_df_agg_w_d.plot(kind='bar')
plt.title('Plot 5 - Average income per day of the week')
# save plot 5 to png
plt.savefig('plot5.png')
plt.show()
################################################# Plot 5 - clickhouse data #################################################

query_5 = query_5 = f'''
WITH
    '{start_date}' AS start_date,
    '{end_date}' AS end_date,

    -- Aggregate total price by date
    sales_by_date AS (
        SELECT
            date,
            sum(total_price) AS total_price
        FROM
            silver_badim.sales
        WHERE
            date >= start_date AND date <= end_date
        GROUP BY
            date
    ),

    -- Map day of the week to names
    sales_with_day_name AS (
        SELECT
            date,
            total_price,
            toDayOfWeek(date) AS day_of_week_num,
            CASE
                WHEN toDayOfWeek(date) = 1 THEN 'Monday'
                WHEN toDayOfWeek(date) = 2 THEN 'Tuesday'
                WHEN toDayOfWeek(date) = 3 THEN 'Wednesday'
                WHEN toDayOfWeek(date) = 4 THEN 'Thursday'
                WHEN toDayOfWeek(date) = 5 THEN 'Friday'
                WHEN toDayOfWeek(date) = 6 THEN 'Saturday'
                WHEN toDayOfWeek(date) = 7 THEN 'Sunday'
            END AS day_of_week
        FROM
            sales_by_date
    )

-- Calculate average total price by day of the week
SELECT
    day_of_week,
    avg(total_price) AS avg_total_price
FROM
    sales_with_day_name
GROUP BY
    day_of_week,
    day_of_week_num
ORDER BY
    day_of_week_num
'''



plot_5_data_clickhouse = client.query_df(query_5)
# fill 0 from sunday to saturday if not exist
plot_5_data_clickhouse = plot_5_data_clickhouse.set_index('day_of_week').reindex(
    ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']).fillna(0)
print("plot 5 data clickhouse", plot_5_data_clickhouse)
plt.bar(plot_5_data_clickhouse.index, plot_5_data_clickhouse['avg_total_price'])
plt.title('Plot 5.1 - Average income per day of the week clickhouse')
plt.show()
##############################################################################################################################

# Plot 6 - Total sales per day of the month
# agg sum of total_price per day first do sum per date and then do mean per day of the month
sales_df_agg_d_s = sales_df.groupby('date').agg({'total_price': 'sum'}).reset_index()
sales_df_agg_d_s['day_of_month'] = sales_df_agg_d_s['date'].dt.day
sales_df_agg_m_d = sales_df_agg_d_s.groupby('day_of_month').agg({'total_price': 'mean'}).fillna(0)
print("plot 6 data", sales_df_agg_m_d)
sales_df_agg_m_d.plot(kind='bar')
plt.title('Plot 6 - Average income per day of the month')
# save plot 6 to png
plt.savefig('plot6.png')
plt.show()

################################################# Plot 6 - clickhouse data #################################################
query_6 = f'''
WITH
    '{start_date}' AS start_date,
    '{end_date}' AS end_date,

    -- Aggregate total price by date
    sales_by_date AS (
        SELECT
            date AS date,
            SUM(total_price) AS total_price
        FROM
            silver_badim.sales
        WHERE
            toDate(date) >= start_date AND toDate(date) <= end_date
        GROUP BY
            date
    ),

    -- Extract the day of the month as string
    sales_with_day_of_month AS (
        SELECT
            date,
            total_price,
            toString(toDayOfMonth(toDate(date))) AS day_of_month_str
        FROM
            sales_by_date
    )

-- Calculate average total price by day of the month
SELECT
    day_of_month_str AS day_of_month,
    AVG(total_price) AS avg_total_price
FROM
    sales_with_day_of_month
GROUP BY
    day_of_month
ORDER BY
    day_of_month
'''

plot_6_data_clickhouse = client.query_df(query_6)
plot_6_data_clickhouse['day_of_month'] = plot_6_data_clickhouse['day_of_month'].astype(int)
plot_6_data_clickhouse = plot_6_data_clickhouse.set_index('day_of_month').sort_index()
print("plot 6 data clickhouse", plot_6_data_clickhouse)
# plot
plt.bar(plot_6_data_clickhouse.index, plot_6_data_clickhouse['avg_total_price'])
plt.title('Plot 6.1 - Average income per day of the month clickhouse')
plt.show()

##############################################################################################################################


# Plot 7 - Total sales per month
# agg sum of total_price per month

# i want month - 1 , month - 2 , month - 3
# will be just month 1,2,3,4,5,6,7,8,9,10,11,12
sales_df['month'] = sales_df['date'].dt.month
sales_df_agg_m_s = sales_df.groupby('month').agg({'total_price': 'mean'}).fillna(0)
print("plot 7 data", sales_df_agg_m_s)
sales_df_agg_m_s.plot(kind='bar')
plt.title('plot 7 - Total income per month')
# save plot 7 to png
plt.savefig('plot7.png')
plt.show()
################################################# Plot 7 - clickhouse data #################################################

query_7 = f'''WITH
    toDate('{start_date}') AS start_date,
    toDate('{end_date}') AS end_date,

    -- Aggregate total price by month
    sales_by_month AS (
        SELECT
            toMonth(toDate(date)) AS month,
            AVG(total_price) AS avg_total_price
        FROM
            silver_badim.sales
        WHERE
            toDate(date) >= start_date AND toDate(date) <= end_date
        GROUP BY
            toMonth(toDate(date))
    )

-- Select the results
SELECT
    month,
    avg_total_price
FROM
    sales_by_month
ORDER BY
    month
'''

plot_7_data_clickhouse = client.query_df(query_7)
plot_7_data_clickhouse['month'] = plot_7_data_clickhouse['month'].astype(int)
plot_7_data_clickhouse = plot_7_data_clickhouse.set_index('month').sort_index()
print("plot 7 data clickhouse", plot_7_data_clickhouse)

plt.bar(plot_7_data_clickhouse.index, plot_7_data_clickhouse['avg_total_price'])
plt.title('Plot 7.1 - Total income per month clickhouse')
plt.show()



##############################################################################################################################

# Plot 8 - i want plot time series of all items that contain 'עור' in item_desc and item that contain 'בד' in item_desc, 4 lines in 1 plot
# i want 4 colors 1 for 'עןר' and no 'בד',   and 1 for 'בד' and no 'עור' and 1 for both and 1 for none
def categorize_item_desc(desc):
    if 'עור' in desc and 'בד' not in desc:
        return 'רק עור'
    elif 'בד' in desc and 'עור' not in desc:
        return 'רק בד'
    elif 'עור' in desc and 'בד' in desc:
        return 'בד וגם עור'
    else:
        return 'לא בד ולא עור'

# Convert 'date' to datetime
sales_df['date'] = pd.to_datetime(sales_df['date'])

# Apply the category function
sales_df['category'] = sales_df['item_desc'].apply(categorize_item_desc)

# Pivot the data to get total price per day per category
pivot_df = sales_df.pivot_table(index='date', columns='category', values='total_price', aggfunc='sum', fill_value=0)

# Resample to fill in missing dates and frequency
pivot_df = pivot_df.resample(agg_date_update).sum()

# Plotting
fig, ax = plt.subplots()
colors = ['blue', 'green', 'red', 'orange']
categories = ['רק עור', 'רק בד', 'בד וגם עור', 'לא בד ולא עור']

for category, color in zip(categories, colors):
    pivot_df[category].plot(kind='line', ax=ax, color=color, label=category)
print("plot 8 data", pivot_df)
plt.title('Plot 8 - Total Price per Day for Different Categories')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Price')

# Save the plot
plt.savefig('plot8.png')
plt.show()
################################################# Plot 8 - clickhouse data #################################################

query_8 = f"""
WITH
    toDate('{start_date}') AS start_date,
    toDate('{end_date}') AS end_date,

    -- Define the categories based on item description
    categorized_sales AS (
        SELECT
            toDate(date) AS date,
            CASE
                WHEN positionUTF8(item_desc, 'עור') > 0 AND positionUTF8(item_desc, 'בד') = 0 THEN 'only_leather'
                WHEN positionUTF8(item_desc, 'בד') > 0 AND positionUTF8(item_desc, 'עור') = 0 THEN 'only_fabric'
                WHEN positionUTF8(item_desc, 'עור') > 0 AND positionUTF8(item_desc, 'בד') > 0 THEN 'leather_and_fabric'
                ELSE 'neither_leather_nor_fabric'
            END AS category,
            total_price
        FROM
            silver_badim.sales
        WHERE
            toDate(date) >= start_date AND toDate(date) <= end_date
    ),

    -- Aggregate total price per day per category
    aggregated_sales AS (
        SELECT
            date,
            category,
            SUM(total_price) AS total_price
        FROM
            categorized_sales
        GROUP BY
            date, category
    )

-- Final selection to get results in a pivot-like format
SELECT
    date,
    sumIf(total_price, category = 'only_leather') AS only_leather,
    sumIf(total_price, category = 'only_fabric') AS only_fabric,
    sumIf(total_price, category = 'leather_and_fabric') AS leather_and_fabric,
    sumIf(total_price, category = 'neither_leather_nor_fabric') AS neither_leather_nor_fabric
FROM
    aggregated_sales
GROUP BY
    date
ORDER BY
    date
"""

plot_8_data_clickhouse = client.query_df(query_8)
# fill missing dates with 0
plot_8_data_clickhouse = plot_8_data_clickhouse.set_index('date').reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
# rename
plot_8_data_clickhouse = plot_8_data_clickhouse.rename(columns={'only_leather': 'רק עור', 'only_fabric': 'רק בד',
                                                                'leather_and_fabric': 'בד וגם עור',
                                                                'neither_leather_nor_fabric': 'לא בד ולא עור'})

print("plot 8 data clickhouse", plot_8_data_clickhouse)

# plot
fig, ax = plt.subplots()
colors = ['blue', 'green', 'red', 'orange']
categories = ['רק עור', 'רק בד', 'בד וגם עור', 'לא בד ולא עור']
for category, color in zip(categories, colors):
    pivot_df[category].plot(kind='line', ax=ax, color=color, label=category)
plt.title('Plot 8.1 - Total Price per Day for Different Categories clickhouse')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Price')
plt.show()

##############################################################################################################################


# plot 9 - do same of 8 just for color and not for item_desc add
def categorize_item_color(item):
    # Check if the item description contains a color and just this color
    if 'כחול' in item and 'חום' not in item and 'אדום' not in item and 'ירוק' not in item and 'צהוב' not in item and 'שחור' not in item and 'לבן' not in item and 'אפור' not in item:
        return 'blue'
    elif 'חום' in item and 'כחול' not in item and 'אדום' not in item and 'ירוק' not in item and 'צהוב' not in item and 'שחור' not in item and 'לבן' not in item and 'אפור' not in item:
        return 'brown'
    elif 'אדום' in item and 'כחול' not in item and 'חום' not in item and 'ירוק' not in item and 'צהוב' not in item and 'שחור' not in item and 'לבן' not in item and 'אפור' not in item:
        return 'red'
    elif 'ירוק' in item and 'כחול' not in item and 'חום' not in item and 'אדום' not in item and 'צהוב' not in item and 'שחור' not in item and 'לבן' not in item and 'אפור' not in item:
        return 'green'
    elif 'צהוב' in item and 'כחול' not in item and 'חום' not in item and 'אדום' not in item and 'ירוק' not in item and 'שחור' not in item and 'לבן' not in item and 'אפור' not in item:
        return 'yellow'
    elif 'שחור' in item and 'כחול' not in item and 'חום' not in item and 'אדום' not in item and 'ירוק' not in item and 'צהוב' not in item and 'לבן' not in item and 'אפור' not in item:
        return 'black'
    elif 'לבן' in item and 'כחול' not in item and 'חום' not in item and 'אדום' not in item and 'ירוק' not in item and 'צהוב' not in item and 'שחור' not in item and 'אפור' not in item:
        return 'white'
    elif 'אפור' in item and 'כחול' not in item and 'חום' not in item and 'אדום' not in item and 'ירוק' not in item and 'צהוב' not in item and 'שחור' not in item and 'לבן' not in item:
        return 'gray'

    # more then 1 color in item_desc
    elif  'כחול' in item or 'חום' in item or 'אדום' in item or 'ירוק' in item or 'צהוב' in item or 'שחור' in item or 'לבן' in item or 'אפור' in item:
        return 'more then 1 color'
    else:
        return 'other'

# Add the new color column
sales_df['color'] = sales_df['item_desc'].apply(categorize_item_color)
# grop by   color and sum total_price

# Plot the time series for each color category

colors = ['blue', 'brown', 'red', 'green', 'yellow', 'black', 'white', 'gray', 'more then 1 color', 'other']
## add backround color to plot
# Apply the color categorization function
sales_df['color'] = sales_df['item_desc'].apply(categorize_item_color)

# Create a pivot table to aggregate total prices by date and color
pivot_df = sales_df.pivot_table(index='date', columns='color', values='total_price', aggfunc='sum', fill_value=0)

# Resample the pivot table to the desired frequency
pivot_df = pivot_df.resample(agg_date_update).sum()

# Plotting
fig, ax = plt.subplots()
ax.set_facecolor('lightgrey')

# Define colors for plotting
plot_colors = {
    'blue': 'blue',
    'brown': 'brown',
    'red': 'red',
    'green': 'green',
    'yellow': 'yellow',
    'black': 'black',
    'white': 'white',
    'gray': 'gray',
    'more than 1 color': 'purple',
    'other': 'pink'
}

# Plot each color category
for color in pivot_df.columns:
    pivot_df[color].plot(kind='line', ax=ax, color=plot_colors.get(color, 'black'), label=color)
# show data
print("plot 9 data", pivot_df)
plt.title('Plot 9 - Total income per day for different colors in description')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Price')
# Save plot 9 to png
plt.savefig('plot9.png')
plt.show()
######################################## Plot 9 - clickhouse data ########################################################
query_9 = f"""WITH
    toDate('{start_date}') AS start_date,
    toDate('{end_date}') AS end_date,

    -- Define the categories based on item description
    categorized_sales AS (
        SELECT
            toDate(date) AS date,
            CASE
                WHEN positionUTF8(item_desc, 'כחול') > 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'לבן') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'blue'
                WHEN positionUTF8(item_desc, 'חום') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'לבן') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'brown'
                WHEN positionUTF8(item_desc, 'אדום') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'לבן') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'red'
                WHEN positionUTF8(item_desc, 'ירוק') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'לבן') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'green'
                WHEN positionUTF8(item_desc, 'צהוב') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'לבן') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'yellow'
                WHEN positionUTF8(item_desc, 'שחור') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'לבן') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'black'
                WHEN positionUTF8(item_desc, 'לבן') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'אפור') = 0 THEN 'white'
                WHEN positionUTF8(item_desc, 'אפור') > 0 AND positionUTF8(item_desc, 'כחול') = 0 AND positionUTF8(item_desc, 'חום') = 0 AND positionUTF8(item_desc, 'אדום') = 0 AND positionUTF8(item_desc, 'ירוק') = 0 AND positionUTF8(item_desc, 'צהוב') = 0 AND positionUTF8(item_desc, 'שחור') = 0 AND positionUTF8(item_desc, 'לבן') = 0 THEN 'gray'
                WHEN positionUTF8(item_desc, 'כחול') > 0 OR positionUTF8(item_desc, 'חום') > 0 OR positionUTF8(item_desc, 'אדום') > 0 OR positionUTF8(item_desc, 'ירוק') > 0 OR positionUTF8(item_desc, 'צהוב') > 0 OR positionUTF8(item_desc, 'שחור') > 0 OR positionUTF8(item_desc, 'לבן') > 0 OR positionUTF8(item_desc, 'אפור') > 0 THEN 'more than 1 color'
                ELSE 'other'
            END AS color,
            total_price
        FROM
            silver_badim.sales
        WHERE
            toDate(date) >= start_date AND toDate(date) <= end_date
    ),

    -- Aggregate total price per day per color
    aggregated_sales AS (
        SELECT
            date,
            color,
            SUM(total_price) AS total_price
        FROM
            categorized_sales
        GROUP BY
            date, color
    )

-- Final selection to get results in a pivot-like format
SELECT
    date,
    sumIf(total_price, color = 'blue') AS blue,
    sumIf(total_price, color = 'brown') AS brown,
    sumIf(total_price, color = 'red') AS red,
    sumIf(total_price, color = 'green') AS green,
    sumIf(total_price, color = 'yellow') AS yellow,
    sumIf(total_price, color = 'black') AS black,
    sumIf(total_price, color = 'white') AS white,
    sumIf(total_price, color = 'gray') AS gray,
    sumIf(total_price, color = 'more than 1 color') AS more_than_1_color,
    sumIf(total_price, color = 'other') AS other
FROM
    aggregated_sales
GROUP BY
    date
ORDER BY
    date"""

plot_9_data_clickhouse = client.query_df(query_9)
plot_9_data_clickhouse = plot_9_data_clickhouse.set_index('date').astype(float)
# fill missing dates with 0
plot_9_data_clickhouse = plot_9_data_clickhouse.reindex(pd.date_range(start=start_date, end=end_date)).fillna(0)
plot_9_data_clickhouse = plot_9_data_clickhouse.resample(agg_date_update).sum()
print("plot 9 data clickhouse", plot_9_data_clickhouse)
# plot
fig, ax = plt.subplots()
ax.set_facecolor('lightgrey')
# Define colors for plotting
plot_colors = {
    'blue': 'blue',
    'brown': 'brown',
    'red': 'red',
    'green': 'green',
    'yellow': 'yellow',
    'black': 'black',
    'white': 'white',
    'gray': 'gray',
    'more than 1 color': 'purple',
    'other': 'pink'
}
# Plot each color category
for color in plot_9_data_clickhouse.columns:
    plot_9_data_clickhouse[color].plot(kind='line', ax=ax, color=plot_colors.get(color, 'black'), label=color)
plt.title('Plot 9.1 - Total income per day for different colors in description clickhouse')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Total Price')
plt.show()


##############################################################################################################################



colors = ['blue', 'brown', 'red', 'green', 'yellow', 'black', 'white', 'gray', 'more then 1 color', 'other']
## add backround color to plot





# plot 10 - do pie chart of sum of total_price per color
# the colors will be the labels and the total_price will be the values
# add label to pie chart
colors_pie = {'blue': 'blue', 'brown': 'brown', 'red': 'red', 'green': 'green', 'yellow': 'yellow', 'black': 'black',
              'white': 'white', 'gray': 'gray', 'more then 1 color': 'purple', 'other': 'pink'}
sum_of_total_price_per_color = sales_df.groupby('color').agg({'total_price': 'sum'}).reset_index()

# Custom colors
custom_colors = [colors_pie[color] for color in sum_of_total_price_per_color['color']]

# Plot
plt.figure(figsize=(10, 7))
plt.pie(sum_of_total_price_per_color['total_price'], labels=sum_of_total_price_per_color['color'], colors=custom_colors, autopct='%1.1f%%')
print( "plot 10 data", sum_of_total_price_per_color)
plt.title('Plot 10 - Total Price by Color')
# save plot 10 to png
plt.savefig('plot10.png')
plt.show()

################################################# Plot 10 - clickhouse data #################################################
query_10 = f"""WITH
    -- Define the categories based on item description
    categorized_sales AS (
        SELECT
            CASE
                WHEN positionUTF8(item_desc, 'כחול') > 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'blue'
                WHEN positionUTF8(item_desc, 'חום') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'brown'
                WHEN positionUTF8(item_desc, 'אדום') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'red'
                WHEN positionUTF8(item_desc, 'ירוק') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'green'
                WHEN positionUTF8(item_desc, 'צהוב') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'yellow'
                WHEN positionUTF8(item_desc, 'שחור') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'black'
                WHEN positionUTF8(item_desc, 'לבן') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'אפור') = 0 THEN 'white'
                WHEN positionUTF8(item_desc, 'אפור') > 0 AND 
                     positionUTF8(item_desc, 'כחול') = 0 AND 
                     positionUTF8(item_desc, 'חום') = 0 AND 
                     positionUTF8(item_desc, 'אדום') = 0 AND 
                     positionUTF8(item_desc, 'ירוק') = 0 AND 
                     positionUTF8(item_desc, 'צהוב') = 0 AND 
                     positionUTF8(item_desc, 'שחור') = 0 AND 
                     positionUTF8(item_desc, 'לבן') = 0 THEN 'gray'
                WHEN positionUTF8(item_desc, 'כחול') > 0 OR 
                     positionUTF8(item_desc, 'חום') > 0 OR 
                     positionUTF8(item_desc, 'אדום') > 0 OR 
                     positionUTF8(item_desc, 'ירוק') > 0 OR 
                     positionUTF8(item_desc, 'צהוב') > 0 OR 
                     positionUTF8(item_desc, 'שחור') > 0 OR 
                     positionUTF8(item_desc, 'לבן') > 0 OR 
                     positionUTF8(item_desc, 'אפור') > 0 THEN 'more than 1 color'
                ELSE 'other'
            END AS color,
            SUM(total_price) AS total_price
        FROM
            silver_badim.sales
        GROUP BY
            color
    )

-- Aggregate total price per color
SELECT
    color,
    SUM(total_price) AS total_price
FROM
    categorized_sales
GROUP BY
    color
ORDER BY
    color
"""

plot_10_data_clickhouse = client.query_df(query_10)
print("plot 10 data clickhouse", plot_10_data_clickhouse)

# plot
# Custom colors
custom_colors = [colors_pie[color] for color in sum_of_total_price_per_color['color']]
# Plot
plt.figure(figsize=(10, 7))
plt.pie(plot_10_data_clickhouse['total_price'], labels=plot_10_data_clickhouse['color'], colors=custom_colors, autopct='%1.1f%%')
plt.title('Plot 10.1 - Total Price by Color clickhouse')
# save plot 10 to png
plt.show()


##############################################################################################################################

# Plot 11 - scatter plot of sales and total_price tick will be 0.1 log scale
sales_df11 = sales_df[['sales', 'total_price', 'date']].groupby('date').sum().reset_index()
# fill missing dates with 0
sales_df11 = sales_df11.set_index('date').reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
# resample the data
sales_df11 = sales_df11.resample(agg_date_update).sum()
plt.scatter(sales_df11['sales'], sales_df11['total_price'], s=1)
print ("plot 11 data", sales_df11[['sales', 'total_price']])
plt.title('Plot 11 - Scatter plot of sales and total price')
plt.xlabel('Sales')
plt.ylabel('Total Price')
# save plot 11 to png
plt.savefig('plot11.png')
plt.show()
######################################### Plot 11 - clickhouse data ########################################################


query_11 = f"""SELECT
    toDate(date) AS date,
    SUM(sales) AS sales,
    SUM(total_price) AS total_price
FROM
    silver_badim.sales
GROUP BY
    date
ORDER BY
    date
"""

plot_11_data_clickhouse = client.query_df(query_11)
plot_11_data_clickhouse = plot_11_data_clickhouse.set_index('date').reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
# resample the data
sales_df11 = sales_df11.resample(agg_date_update).sum()


print("plot 11 data clickhouse", plot_11_data_clickhouse)

# plot
plt.scatter(plot_11_data_clickhouse['sales'], plot_11_data_clickhouse['total_price'], s=1)
plt.title('Plot 11.1 - Scatter plot of sales and total price clickhouse')
plt.xlabel('Sales')
plt.ylabel('Total Price')
plt.show()

##############################################################################################################################
supply_orders_df = client.query_df(f'''SELECT * FROM silver_badim.supply_orders
                            WHERE date >= '{start_date}' AND date <= '{end_date}'  ''')
# save copy of csv
supply_orders_df.to_csv('supply_orders.csv')
# add colunm convert to ILS from USD and from EUR {'ILS': 1, 'USD': 3.7, 'EUR': 4.0}
print(supply_orders_df.info())
supply_orders_df['total_price'] = supply_orders_df['total_price'].astype(float)

supply_orders_df['total_price_ILS'] = supply_orders_df['total_price'] * supply_orders_df['coin'].map({'ILS': 1, 'USD': 3.7, 'EUR': 4.0})
print(supply_orders_df)

# Plot 12 - show pie chart of total_price_ILS of status column sum of total_price_ILS the labels are in hebrew add the total_price_ILS in the pie chart
sum_of_total_price_ILS_per_status = supply_orders_df.groupby('status').agg({'total_price_ILS': 'sum'}).reset_index()
print("plot 12 data", sum_of_total_price_ILS_per_status)
# Format the labels to include both the status and the total price
labels = [f"{status[::-1]} - {price:.2f} ILS" for status, price in zip(sum_of_total_price_ILS_per_status['status'], sum_of_total_price_ILS_per_status['total_price_ILS'])]

plt.figure(figsize=(10, 7))
plt.pie(sum_of_total_price_ILS_per_status['total_price_ILS'], labels=labels, autopct='%1.1f%%')
plt.title('Plot 12 - ' + (u'סה"כ מחיר בש"ח לפי סטטוס')[::-1])
# save plot 12 to png
plt.savefig('plot12.png')
plt.show()
####################################### Plot 12 - clickhouse data ########################################################
query_12 = f"""
SELECT
    status,
    SUM(
        total_price * 
        CASE 
            WHEN coin = 'USD' THEN 3.7 
            WHEN coin = 'EUR' THEN 4.0 
            ELSE 1 
        END
    ) AS total_price_ILS
FROM
    silver_badim.supply_orders
WHERE
    date >= '{start_date}' AND date <= '{end_date}'
GROUP BY
    status
ORDER BY
    status"""

plot_12_data_clickhouse = client.query_df(query_12)
print("plot 12 data clickhouse", plot_12_data_clickhouse)

# plot
# Format the labels to include both the status and the total price
labels = [f"{status[::-1]} - {price:.2f} ILS" for status, price in zip(plot_12_data_clickhouse['status'], plot_12_data_clickhouse['total_price_ILS'])]

plt.figure(figsize=(10, 7))
plt.pie(plot_12_data_clickhouse['total_price_ILS'], labels=labels, autopct='%1.1f%%')
plt.title('Plot 12.1 - ' + (u'סה"כ מחיר בש"ח לפי סטטוס')[::-1])
plt.show()



##############################################################################################################################
supply_orders_df['date'] = pd.to_datetime(supply_orders_df['date'])
sales_df_agg_d_s = sales_df.groupby('date').agg({'total_price': 'sum'}).reset_index()
supply_orders_df_agg_d_s = supply_orders_df.groupby('date').agg({'total_price_ILS': 'sum'}).reset_index()
supply_orders_df_agg_d_s = supply_orders_df_agg_d_s[supply_orders_df_agg_d_s['date'] >= '2024-04-01']
print(supply_orders_df_agg_d_s)
print(sales_df_agg_d_s)
# Merge the two dataframes
merged_df = pd.merge(sales_df_agg_d_s, supply_orders_df_agg_d_s, on='date', how='outer').fillna(0)
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df = merged_df.sort_values('date')
merged_df['diff'] = merged_df['total_price'] - merged_df['total_price_ILS']
# Plot 13 - Total Price (Sales) vs Total Price (Supply Orders) per Date
print("plot 13 data", merged_df[['date', 'total_price', 'total_price_ILS', 'diff']])
plt.figure(figsize=(14, 7))
plt.plot(merged_df['date'], merged_df['total_price'], label='Total Price (Sales)', color='blue')
plt.plot(merged_df['date'], merged_df['total_price_ILS'], label='Total Price (Supply Orders)', color='red')
plt.plot(merged_df['date'], merged_df['diff'], label='Difference(income-expense)', color='green')
plt.axvline(x=target_date, color='r', linestyle='--', label='from sap to priority')
plt.xticks(rotation=45)
plt.title('Plot 13 - Total Price (Sales) vs Total Price (Supply Orders) per Date')
plt.legend()
# save plot 13 to png
plt.savefig('plot13.png')
plt.show()
######################################### Plot 13 - clickhouse data ########################################################

query_13 = f"""WITH
    -- Aggregate sales by date
    sales_agg AS (
        SELECT
            toDate(date) AS date,
            SUM(total_price) AS total_price
        FROM
            silver_badim.sales
        WHERE
            toDate(date) >= '2024-04-01'    
        GROUP BY
            date
    ),

    -- Aggregate supply_orders by date and convert to ILS
    supply_orders_agg AS (
        SELECT
            toDate(date) AS date,
            SUM(
                total_price * 
                CASE 
                    WHEN coin = 'USD' THEN 3.7 
                    WHEN coin = 'EUR' THEN 4.0 
                    ELSE 1 
                END
            ) AS total_price_ILS
        FROM
            silver_badim.supply_orders
        WHERE
            toDate(date) >= '2024-04-01'
        GROUP BY
            date
    )

-- Merge the two results and calculate the difference
SELECT
    COALESCE(sales_agg.date, supply_orders_agg.date) AS date,
    COALESCE(sales_agg.total_price, 0) AS total_price,
    COALESCE(supply_orders_agg.total_price_ILS, 0) AS total_price_ILS,
    COALESCE(sales_agg.total_price, 0) - COALESCE(supply_orders_agg.total_price_ILS, 0) AS diff
FROM
    sales_agg
FULL OUTER JOIN
    supply_orders_agg
ON
    sales_agg.date = supply_orders_agg.date
-- filter by date from start_date until end_date
WHERE
    date >= toDate('{start_date}') AND date <= toDate('{end_date}')
    
ORDER BY
    date"""

plot_13_data_clickhouse = client.query_df(query_13)
print("plot 13 data clickhouse", plot_13_data_clickhouse)

# Plot
plt.figure(figsize=(14, 7))
plt.plot(plot_13_data_clickhouse['date'], plot_13_data_clickhouse['total_price'], label='Total Price (Sales)', color='blue')
plt.plot(plot_13_data_clickhouse['date'], plot_13_data_clickhouse['total_price_ILS'], label='Total Price (Supply Orders)', color='red')
plt.plot(plot_13_data_clickhouse['date'], plot_13_data_clickhouse['diff'], label='Difference(income-expense)', color='green')
plt.axvline(x=target_date, color='r', linestyle='--', label='from sap to priority')
plt.xticks(rotation=45)
plt.title('Plot 13.1 - Total Price (Sales) vs Total Price (Supply Orders) per Date clickhouse')
plt.legend()
plt.show()



##############################################################################################################################
orders_df = client.query_df(f'''
    SELECT 
        o.*, 
        c.country AS cust_country, 
        c.address AS cust_address
    FROM 
        silver_badim.orders o
    LEFT JOIN 
        silver_badim.customers c
    ON 
        o.cust_id = c.cust_id
    WHERE 
        o.date >= '{start_date}' AND o.date <= '{end_date}' ''')
print(orders_df)
# add column that will contain only year and month in format '2021-01'
# plot 14 - hist of total_price
orders_df['total_price_with_discount'] = orders_df['total_price_with_discount'].astype(float)
orders_df['total_price_with_discount'].plot(kind='hist', bins=np.arange(0, 2500, 10))
# print data
print("plot 14 data", orders_df[['total_price_with_discount']])
plt.title('Plot 14 - Total income from orders Histogram')
# save plot 14 to png
plt.savefig('plot14.png')
plt.show()
######################################### Plot 14 - clickhouse data ########################################################
query_14 = f"""SELECT 
    CAST(o.total_price_with_discount AS Float64) AS total_price_with_discount
 FROM 
        silver_badim.orders o
    LEFT JOIN 
        silver_badim.customers c
    ON 
        o.cust_id = c.cust_id
    WHERE 
        o.date >= '{start_date}' AND o.date <= '{end_date}'"""

plot_14_data_clickhouse = client.query_df(query_14)
print("plot 14 data clickhouse", plot_14_data_clickhouse)
    
# plot
plot_14_data_clickhouse['total_price_with_discount'].plot(kind='hist', bins=np.arange(0, 2500, 10))
plt.title('Plot 14.1 - Total income from orders Histogram clickhouse')
plt.show()
##############################################################################################################################
# Plot 15 - show the pie of sum of total_price_with_discount per country
sum_of_total_price_with_discount_per_country = orders_df.groupby('cust_country').agg({'total_price_with_discount': 'sum'}).reset_index()
# Format the labels to include both the country and the total price
labels = [f"{country} - {price:.2f} ILS" for country, price in zip(sum_of_total_price_with_discount_per_country['cust_country'], sum_of_total_price_with_discount_per_country['total_price_with_discount'])]
print("plot 15 data", sum_of_total_price_with_discount_per_country[['total_price_with_discount']])
plt.figure(figsize=(10, 7))
plt.pie(sum_of_total_price_with_discount_per_country['total_price_with_discount'], labels=labels, autopct='%1.1f%%')
plt.title('Plot 15 - Total Price with Discount by Country')
# save plot 15 to png
plt.savefig('plot15.png')
plt.show()
######################################### Plot 15 - clickhouse data ########################################################
query_15 = f"""SELECT 
    c.country AS cust_country, 
    SUM(CAST(o.total_price_with_discount AS Float64)) AS total_price_with_discount
FROM 
    silver_badim.orders o
LEFT JOIN 
    silver_badim.customers c
ON 
    o.cust_id = c.cust_id
WHERE 
    o.date >= '{start_date}' AND o.date <= '{end_date}'
GROUP BY 
    cust_country
ORDER BY 
    total_price_with_discount DESC"""

plot_15_data_clickhouse = client.query_df(query_15)
print("plot 15 data clickhouse", plot_15_data_clickhouse)
## add backround color to plot
# plot
# Format the labels to include both the country and the total price
labels = [f"{country} - {price:.2f} ILS" for country, price in zip(plot_15_data_clickhouse['cust_country'], plot_15_data_clickhouse['total_price_with_discount'])]
plt.figure(figsize=(10, 7))
plt.pie(plot_15_data_clickhouse['total_price_with_discount'], labels=labels, autopct='%1.1f%%')
plt.title('Plot 15.1 - Total Price with Discount by Country clickhouse')
plt.show()
##############################################################################################################################



# Plot 16 - show how many orders by date time series plot
orders_df['date'] = pd.to_datetime(orders_df['date'])
orders_df_agg_d = orders_df.groupby('date').agg({'order_id': 'count'})
orders_df_agg_d = orders_df_agg_d.reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).reset_index().rename(columns={'index': 'date'})
# Plot the time series for the number of orders per day
print("plot 16 data", orders_df_agg_d[['date', 'order_id']])
orders_df_agg_d.plot(kind='line', x='date', y='order_id')
plt.title('Plot 16 - Number of Orders per Day')
# save plot 16 to png
plt.savefig('plot16.png')
plt.show()

######################################### Plot 16 - clickhouse data ########################################################
query_16 = f"""WITH
    -- Aggregate orders by date
    orders_agg AS (
        SELECT
            toDate(date) AS date,
            COUNT(order_id) AS order_count
        FROM
            silver_badim.orders
        WHERE
            date >= '{start_date}' AND date <= '{end_date}'
        GROUP BY
            date
    ),

    -- Generate a date range from the specified start_date to end_date
    date_range AS (
        SELECT
            arrayJoin(
                range(
                    toUnixTimestamp(toDate('{start_date}')),
                    toUnixTimestamp(toDate('{end_date}')) + 86400,
                    86400
                )
            ) AS ts,
            toDate(ts) AS date
    )

-- Merge the results and fill missing dates with zero order counts
SELECT
    d.date AS date,
    COALESCE(o.order_count, 0) AS order_count
FROM
    date_range d
LEFT JOIN
    orders_agg o ON d.date = o.date
ORDER BY
    date"""

plot_16_data_clickhouse = client.query_df(query_16)
print("plot 16 data clickhouse", plot_16_data_clickhouse)

# plot the time series for the number of orders per day
plot_16_data_clickhouse.plot(kind='line', x='date', y='order_count')
plt.title('Plot 16.1 - Number of Orders per Day clickhouse')
# save plot 16.1 to png

plt.savefig('plot16.1.png')
plt.show()

window_var = 7
window_trend = 7

# Plotting the trend
sales_df_g_s = sales_df.groupby('date').agg({'total_price': 'sum'}).reset_index()
# fill missing dates with 0
sales_df_g_s = sales_df_g_s.set_index('date').reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
sales_df_g_s['price_variance'] = sales_df_g_s['total_price'].rolling(window=window_var).var()
plt.figure(figsize=(10, 5))

print("plot 17 data", sales_df_g_s[['price_variance']])
# Plotting the variance
plt.figure(figsize=(10, 5))
plt.plot(sales_df_g_s['price_variance'], label='Variance', color='red')
plt.xlabel('Date')
plt.ylabel('Variance')
plt.title('Plot 17 - Income Variance Over Time')
plt.legend()
# save plot 17 to png
plt.savefig('plot17.png')
plt.show()
######################################### Plot 17 - clickhouse data ########################################################
query_17 = f'''
    SELECT
        toDate(date) AS date,
        sum(total_price) AS total_price
    FROM silver_badim.sales
    WHERE toDate(date) >= toDate('{start_date}') AND toDate(date) <= toDate('{end_date}')
    GROUP BY date
    ORDER BY date
'''

plot_17_data_clickhouse = client.query_df(query_17)
plot_17_data_clickhouse['date'] = pd.to_datetime(plot_17_data_clickhouse['date'])
plot_17_data_clickhouse = plot_17_data_clickhouse.set_index('date').reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
plot_17_data_clickhouse['price_variance'] = plot_17_data_clickhouse['total_price'].rolling(window=window_var).var()
# add rolling variance

print("plot 17 data clickhouse", plot_17_data_clickhouse[['price_variance']])

##############################################################################################################################
sales_df_g_s['moving_average'] = sales_df_g_s['total_price'].rolling(window=window_trend).mean()
print("plot 18 data", sales_df_g_s[['moving_average']])
# Plotting the trend
plt.figure(figsize=(10, 5))
plt.plot(sales_df_g_s['total_price'], label='Total Price', color='blue', alpha=0.5)
plt.plot(sales_df_g_s['moving_average'], label='Moving Average (Trend) of {} points'.format(window_trend), color='red')
plt.xlabel('Date')
plt.ylabel('Total Price')
plt.title('Plot 18 - Income Trend Over Time')
plt.legend()
plt.grid(True)

# save plot 18 to png
plt.savefig('plot18.png')
plt.show()

######################################### Plot 18 - clickhouse data ########################################################
query_18 = f'''
    SELECT
        toDate(date) AS date,
        sum(total_price) AS total_price
    FROM silver_badim.sales
    WHERE toDate(date) >= toDate('{start_date}') AND toDate(date) <= toDate('{end_date}')
    GROUP BY date
    ORDER BY date
'''

plot_18_data_clickhouse = client.query_df(query_18)
plot_18_data_clickhouse['date'] = pd.to_datetime(plot_18_data_clickhouse['date'])
plot_18_data_clickhouse = plot_18_data_clickhouse.set_index('date').reindex(pd.date_range(start=start_date, end=end_date)).fillna(0).resample(agg_date_update).sum()
plot_18_data_clickhouse['moving_average'] = plot_18_data_clickhouse['total_price'].rolling(window=window_trend).mean()
print("plot 18 data clickhouse", plot_18_data_clickhouse[['moving_average']])


##############################################################################################################################