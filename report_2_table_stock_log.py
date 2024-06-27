#%%
import os
#import clickhouse_connect
import clickhouse_driver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']
#%%
start_date = '2024-04-01'
end_date = '2024-06-27'
agg_time_freq = 'W'
inv_mov_types = []
type_of_filter = 'item'
list_of_type = ['20004053']
units = []
status_supply_orders = ['שוחרר']
status_regular_orders = ['בוצעה','שולמה']
client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)
params = {'start_date': start_date, 'end_date': end_date}
#%%
def filter_for_query(name_of_column, filter_list):
    if len(filter_list) == 0:
        return ''
    else:
        # add quotes to string values
        formatted_values = ", ".join([f"'{x}'" if isinstance(x, str) else str(x) for x in filter_list])
        return f"AND {name_of_column} IN ({formatted_values})"

def filter_from_right_item_charachters(filter_list, index_of_first_char, lenght_of_chars, name_of_column='item'):
    if len(filter_list) == 0:
        return ''
    else:
        formatted_values = ", ".join([f"'{x}'" if isinstance(x, str) else str(x) for x in filter_list])
        return f"AND SUBSTRING({name_of_column}, {index_of_first_char + 1}, {lenght_of_chars}) IN ({formatted_values})"

def sales_or_income_columns_name(sales_or_income):
    if sales_or_income == 'income':
        return 'total_price'
    elif sales_or_income == 'sales':
        return 'sales'
if agg_time_freq == 'no_agg':
    agg_time_freq = 'D'
date_trunc_func = {
    'D': 'toStartOfDay',
    'W': 'toStartOfWeek',
    'M': 'toStartOfMonth',
    'Q': 'toStartOfQuarter',
    'Y': 'toStartOfYear'
}.get(agg_time_freq, 'toStartOfDay')
def item_cataegory_catalog_or_color_query(name_of_column, filter_list):
    if name_of_column == 'category':
        return filter_from_right_item_charachters(filter_list, 0, 2)
    elif name_of_column == 'catalog':
        return filter_from_right_item_charachters(filter_list, 2, 3)
    elif name_of_column == 'color':
        return filter_from_right_item_charachters(filter_list, 5, 3)
    elif name_of_column == 'item':
        return filter_for_query(name_of_column, filter_list)

query = f'''
WITH
    outgoing AS (
        SELECT
            toStartOfWeek(toDate(update_date)) as agg_date,
            sum(quantity) as total_quantity
        FROM
            silver_badim.stock_log
        WHERE toDate(update_date) >= toDate(%(start_date)s) AND toDate(update_date) <= toDate(%(end_date)s)
    {filter_for_query('unit', units)}
    {item_cataegory_catalog_or_color_query(type_of_filter, list_of_type)}
            AND inv_mov_type in ('משלוחים ללקוח', 'חשבוניות מס', 'חשבוניות מס קבלה', 'דאטה מסאפ')
        GROUP BY
            agg_date
    ),
    returns AS (
        SELECT
            toStartOfWeek(toDate(update_date)) as agg_date,
            sum(quantity) as total_quantity
        FROM
            silver_badim.stock_log
               WHERE toDate(update_date) >= toDate(%(start_date)s) AND toDate(update_date) <= toDate(%(end_date)s)
    {item_cataegory_catalog_or_color_query(type_of_filter, list_of_type) }
    {filter_for_query('unit', units)}

            AND inv_mov_type in ('החזרה מלקוח')
        GROUP BY
            agg_date
    )
SELECT
    o.agg_date,
    o.total_quantity - COALESCE(r.total_quantity, 0) as net_quantity
FROM
    outgoing o
LEFT JOIN
    returns r ON o.agg_date = r.agg_date
ORDER BY
    o.agg_date;

'''
print(query)
df = client.query_dataframe(query, params=params)
df['agg_date'] = pd.to_datetime(df['agg_date'])
# fill missing dates with 0
df = df.set_index('agg_date').resample(agg_time_freq).sum().fillna(0).reset_index()
df['net_quantity'] = np.round(df['net_quantity'], 2)
print(df)
# plot the data
plt.plot(df['agg_date'], df['net_quantity'])
plt.show()

# supply orders
# i want to see per supply_name how many total_price_ILS was ordered
query = f'''
SELECT
    supply_name,
    sum(total_price_ILS) as total_price_ILS
FROM
    silver_badim.supply_orders
WHERE toDate(date) >= toDate(%(start_date)s) AND toDate(date) <= toDate(%(end_date)s)
{filter_for_query('status', status_supply_orders)}

GROUP BY
    supply_name
ORDER BY
    total_price_ILS DESC;
'''
df = client.query_dataframe(query, params=params)
print(df)
# plot Pie chart
plt.pie(df['total_price_ILS'], labels=df['supply_name'], autopct='%1.1f%%')
plt.show()

# do tme series
query = f'''
SELECT
    {date_trunc_func}(toDate(date)) as agg_date,
    sum(total_price_ILS) as total_price_ILS
FROM
    silver_badim.supply_orders
        
WHERE toDate(date) >= toDate(%(start_date)s) AND toDate(date) <= toDate(%(end_date)s)
{filter_for_query('status', status_supply_orders)}

GROUP BY
    agg_date
ORDER BY
    agg_date;
'''
print(query)
df = client.query_dataframe(query, params=params)
df['agg_date'] = pd.to_datetime(df['agg_date'])
# fill missing dates with 0
df = df.set_index('agg_date').resample(agg_time_freq).sum().fillna(0).reset_index()
df['total_price_ILS'] = np.round(df['total_price_ILS'].astype(float), 2)
print(df)
# plot the data
plt.plot(df['agg_date'], df['total_price_ILS'])
plt.show()

query = f'''
SELECT     c.state AS cust_city,    
 SUM(CAST(o.total_price_with_discount AS Float64)) AS total_price_with_discount
 FROM     silver_badim.orders o LEFT JOIN     silver_badim.customers c ON  
    o.cust_id = c.cust_id WHERE     o.status_date >= %(start_date)s 
    AND o.status_date <= %(end_date)s   
     {filter_for_query('o.order_status', status_regular_orders)}
      GROUP BY     cust_city ORDER BY     total_price_with_discount DESC;'''
df = client.query_dataframe(query, params=params)
# i want labels [::-1] to reverse the order
labels =  [f"{city[::-1]} - {income:.2f} ILS" for city, income in zip(df['cust_city'], df['total_price_with_discount'])]


# plot the name
# plot Pie chart
plt.pie(df['total_price_with_discount'], labels=labels, autopct='%1.1f%%')
plt.show()
print(df)

## i want number of unique per date
query = f'''
SELECT
    {date_trunc_func}(toDate(status_date)) as agg_date,
    count(distinct cust_id) as unique_customers
FROM
    silver_badim.orders
        
        
WHERE toDate(status_date) >= toDate(%(start_date)s) AND toDate(status_date) <= toDate(%(end_date)s)
{filter_for_query('order_status', status_regular_orders)}


GROUP BY
    agg_date
ORDER BY

    agg_date;
'''
df = client.query_dataframe(query, params=params)
df['agg_date'] = pd.to_datetime(df['agg_date'])
# fill missing dates with 0
df = df.set_index('agg_date').resample(agg_time_freq).sum().fillna(0).reset_index()
if agg_time_freq == 'M':
    df.index = df.index.astype(str).str[:7]
if agg_time_freq == 'Y':
    df.index = df.index.astype(str).str[:4]
df['unique_customers'] = df['unique_customers'].astype(int)
print(df)
# plot the data
plt.plot(df['agg_date'], df['unique_customers'])
plt.show()
order_status = []
query = f'''
    SELECT sum(total_price) as Total_income
    FROM silver_badim.sales
    WHERE toDate(status_date) >= toDate(%(start_date)s) AND toDate(status_date) <= toDate(%(end_date)s)
    {filter_for_query('order_status', order_status)}
    {filter_for_query('unit', units)}
    {item_cataegory_catalog_or_color_query(type_of_filter, list_of_type)}
'''
df = client.execute(query, params=params)
print(df)
