import json
import pandas as pd

with open('response3.json') as f:
    data = json.load(f)

# shoe head of the json file
print(data.keys())
print(data['value'][0].keys())
orders_data = pd.DataFrame(data['value'])
orders_df = orders_data.drop(columns=['ORDERITEMS_SUBFORM'])

# Prepare a list to hold the data for order items
order_items_list = []

# Loop through each order to extract order items and add 'ORDNAME'
for index, row in orders_data.iterrows():
    for item in row['ORDERITEMS_SUBFORM']:
        item['ORDNAME'] = row['ORDNAME']  # Add 'ORDNAME' from parent order
        order_items_list.append(item)

# Convert list to DataFrame
order_items_df = pd.DataFrame(order_items_list)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(orders_df.iloc[0])
print("!!!!!")
print(order_items_df.iloc[0])
print(orders_df.shape)
print(orders_df.columns)
order_items_df.to_csv('ORDERITEMS.csv', index=False)

print("1000006")
print(order_items_df[order_items_df['ORDNAME'] == '1000006'])
print("!!!!!!111111")
print(orders_df[orders_df['ORDNAME'] == '1000006'])