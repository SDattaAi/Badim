import pandas as pd
import clickhouse_driver
import os
import numpy as np

password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

client = clickhouse_driver.Client(host=host, user=username, password=password, port=port, secure=True)


df = pd.read_csv('SWA_results.csv')
# drop where unique_id iis nan
df = df[df['unique_id'] == '20_category']
# replace NaN with null
#plot info of df
print(df.info())

#df[['SWA_value', 'MAE', 'MAPE']] = df[['SWA_value', 'MAE', 'MAPE']].fillna(value=None)
print(df)
client.execute('INSERT INTO  platinum_badim.SWA_results VALUES', df.to_dict(orient='records'))