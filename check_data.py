import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('sap_clean_data.csv')
# make all floats to int and if not possible dont change in sku column
def try_convert(x):
    try:
        return int(x)
    except ValueError:
        return x
df['sku'] = df['sku'].apply(lambda x: try_convert(x))
df['sku'] = df['sku'].astype('str')
# show all columns
pd.set_option('display.max_columns', None)
print(df.describe())
# show how count 'total_in_line' and 'price*quantity' rows in all df and how many not equal
count = 0
for i in range(len(df)):
    if df['total_in_line'][i] != df['price*quantity'][i]:
        count += 1

print("total_in_line not equal price*quantity: ", count)
print("total_in_line equal price*quantity: ", len(df) - count)
print("df that not eqal in 'total_in_line' and 'price*quantity': ", df [df['total_in_line'] != df['price*quantity']][['total_in_line', 'price*quantity']])

# plot disterbution of quantity
df['quantity'].plot(kind='hist', bins=np.arange(-100, 1000, 1))
#y lim as 200
plt.ylim(0, 12000)
plt.xlim(-50, 200)
plt.title('Quantity Distribution')
plt.show()

df['price_per_unit'].plot(kind='hist', bins=2000)
plt.title('Price per unit Distribution')
plt.xlim(-50, 400)
plt.show()

# plot delivry type distribution
df['delivery_type'].value_counts().plot(kind='bar')
plt.title('Delivery Type Distribution')
plt.show()

df['price*quantity'].plot(kind='hist', bins=2000)
plt.title('Price*Quantity Distribution')
plt.xlim(-1000, 7000)
plt.show()

# plot discount_percentage distribution, bins [0, 100]
df['discount_percentage'].plot(kind='hist', bins=np.arange(0, 101, 1))
plt.title('Discount Percentage Distribution')
plt.xlim(-10, 100)
plt.show()

# plot time series of y is yyyy-mm and x is price*quantity
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')
df.groupby('year_month')['price*quantity'].sum().plot(kind='line')
plt.title('Price*Quantity by Month')
plt.show()



print(df.columns)
print("sku nunique:", df['sku'].nunique())
print("description_sku nunique:", df['description_sku'].nunique())
print("len(df):", len(df))
# show top 50 sku with the most quantity
print(df.groupby('sku')['quantity'].count().sort_values(ascending=False).head(50))
# print top 50 description_sku with the most quantity
print(df.groupby('description_sku')['quantity'].count().sort_values(ascending=False).head(50))


print(df[df['sku'] == '10900001'].groupby('description_sku')['quantity'].count().sort_values(ascending=False).head(50))
# show all df[df['sku'] == '10900001']['description_sku']  that not conatin 'עור'
print(" contain 'עור' in description_sku")
print("number of unique description with sku 10900001: ", df[df['sku'] == '10900001']['description_sku'].nunique())
print(df[df['sku'] == '10900001'][df[df['sku'] == '10900001']['description_sku'].str.contains('עור')]['description_sku'])
print(" not contain 'עור' in description_sku")
print(df[df['sku'] == '10900001'][~df[df['sku'] == '10900001']['description_sku'].str.contains('עור')]['description_sku'])

print("money from '10900001' : ", df[df['sku'] == '10900001']['price*quantity'].sum())
print("money not from '10900001': ", df[df['sku'] != '10900001']['price*quantity'].sum())

# plot money from '10900001' and without from '10900001'
df[df['sku'] == '10900001'].groupby('year_month')['price*quantity'].sum().plot(kind='line')
df[df['sku'] != '10900001'].groupby('year_month')['price*quantity'].sum().plot(kind='line')
plt.title('Price*Quantity by Month')
plt.legend(['10900001', 'not 10900001'])
plt.show()

#do it for years
df['year'] = df['date'].dt.to_period('Y')
df[df['sku'] == '10900001'].groupby('year')['price*quantity'].sum().plot(kind='line')
df[df['sku'] != '10900001'].groupby('year')['price*quantity'].sum().plot(kind='line')
plt.title('Price*Quantity by Year')
plt.legend(['10900001', 'not 10900001'])
plt.show()