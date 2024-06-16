import json
import pandas as pd
import os
import clickhouse_connect

# Load JSON data
with open('logfile.json') as f:
    data = json.load(f)

# Convert JSON data to DataFrame and ensure all columns are strings
df = pd.DataFrame(data['value'])

# Convert 'None' strings to actual None (null) values
df.replace('None', None, inplace=True)

# Ensure all numeric columns have the correct types
numeric_columns = ['TQUANT', 'QUANT', 'UCOST', 'COST', 'USECONDCOST', 'SECONDCOST', 'NUMPACK', 'DOC', 'TRANS', 'KLINE']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Retrieve connection parameters from environment variables
password = os.environ['CLICKHOUSE_PASSWORD']
username = os.environ['CLICKHOUSE_USERNAME']
port = int(os.environ['CLICKHOUSE_PORT'])
host = os.environ['CLICKHOUSE_HOST']

# Print connection details for verification (optional, remove in production)
print(password, username, port, host)

# Connect to ClickHouse
client = clickhouse_connect.get_client(host=host, user=username, password=password, port=port, database='bronze_badim')

# Ensure the DataFrame is converted to a list of dictionaries correctly
records = df.to_dict(orient='records')

# Verify the structure of the records (optional, remove in production)
print(records[:5])  # Print first 5 records for verification

# Insert data into the ClickHouse table
client.insert('bronze_badim.LOGFILE', records)
