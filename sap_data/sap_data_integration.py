import pandas as pd
import clickhouse_driver
import os
raise Exception('Please remove this line and write your own code')
df = pd.read_csv('sap_data_for_integration.csv')
password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']
print(df.head())
df = df.rename(columns={'קוד פריוריטי': 'item' ,  'תיאור פריט': 'desc_he'})
df = df[['item','desc_he', '5', '6', '7', '8', '9', '10', '11', '12', '1', '2', '3', '4']]
df = df.rename(columns={'5': '2023-05-01', '6': '2023-06-01', '7': '2023-07-01', '8': '2023-08-01', '9': '2023-09-01', '10': '2023-10-01', '11': '2023-11-01', '12': '2023-12-01', '1': '2024-01-01', '2': '2024-02-01', '3': '2024-03-01', '4': '2024-04-01'})
print(df.head())
# show nan values for each column
# drop nan values from item column
df = df.dropna(subset=['item'])
# fill nan values with 0
df = df.fillna(0)
print(df.isnull().sum())
# unpivot the table
df = df.melt(id_vars=['item', 'desc_he'], var_name='date', value_name='quantity')
df['inv_mov_type'] = 'דאטה מסאפ'
df['doc_no'] = None
df['cust_id'] = None
df['cust_name'] = None
df['supply_id'] = None
df[['supply_name', 'warehouse_name', 'UCOST', 'COST', 'USECONDCOST', 'SECONDCOST', 'coin', 'to_warehouse_name', 'general_item']] = None
df['supply_name'] = df['date']
df['update_date'] = df['date']
df['cur_date'] = df['date']
df[[ 'trans_id', 'line_id', 'type',]] = None

df[ 'trans_id']  ='0'
client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)

query = '''SELECT 
    item,
    arrayFirst(x -> 1, arraySort(groupArray(unit))) AS unit
FROM 
    silver_badim.stock_log
GROUP BY 
    item;'''

units = client.query_dataframe(query)
df = df.merge(units, on='item', how='left')
df['quantity_in_warehouse'] = 0.0
df = df[['cur_date', 'inv_mov_type', 'doc_no', 'cust_id', 'cust_name', 'supply_id', 'supply_name',
         'warehouse_name', 'item', 'desc_he', 'quantity', 'unit', 'UCOST', 'COST', 'USECONDCOST',
         'SECONDCOST', 'coin', 'to_warehouse_name', 'general_item'
, 'update_date', 'trans_id', 'line_id', 'type']]


df[['UCOST', 'COST','USECONDCOST','SECONDCOST']] = 0.0
df['item'] = df['item'].astype(str)

df = df.where(pd.notnull(df), None)
print(df.head())
print(df.info())
# upload to clickhouse to silver_badim.stock_log
client.execute('''
INSERT INTO silver_badim.stock_log
(cur_date, inv_mov_type, doc_no, cust_id, cust_name, supply_id, supply_name, warehouse_name, item, desc_he, quantity, unit, UCOST, COST, USECONDCOST, SECONDCOST, coin, to_warehouse_name, 
general_item, update_date, trans_id, line_id, type)
VALUES
''', df.to_dict(orient='records'))

