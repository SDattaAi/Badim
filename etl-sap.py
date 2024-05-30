

import matplotlib.pyplot as plt


import pandas as pd


df = pd.read_csv('raw_data.csv')

print(df.head())
print(df.describe())
print(df.columns)

def format_date(date_str):
    # Check if date_str is a string
    if pd.isna(date_str):
        return date_str  # Return NaN as it is

    try:
        day, month, year = date_str.split('/')
        day = day.zfill(2)
        month = month.zfill(2)
        if len(year) == 2:
            year = '20' + year
        elif len(year) == 3:
            year = '2' + year
        return f'{day}/{month}/{year}'
    except ValueError:
        return date_str  # Return the original string if it can't be split properly


df['תאריך אסמכתא'] = df['תאריך אסמכתא'].apply(format_date)
df['תאריך אספקה לשורה'] = df['תאריך אספקה לשורה'].apply(format_date)
print(df['תאריך אסמכתא'])
print("222222")
# count nans in קוד פריט
print(df['קוד פריט'].isna().sum())
# replace nan 'קוד פריט' with 'nan'
df['קוד פריט'] = df['קוד פריט'].fillna('nan')
print(df.info())
# convert to int  in קוד פריט if possible.
def try_convert(x):
    try:
        return str(int(x))
    except ValueError:
        return str(x)
df['קוד פריט'] = df['קוד פריט'].apply(lambda x: try_convert(x))

df['תאריך אסמכתא'] = pd.to_datetime(df['תאריך אסמכתא'], dayfirst=True)
df['תאריך אספקה לשורה'] = pd.to_datetime(df['תאריך אספקה לשורה'], dayfirst=True)
df = df.rename(columns={'תאריך אסמכתא': 'date',
                        'תאריך אספקה לשורה': 'date_supply',
                        'קוד פריט': 'sku',
                        'כמות': 'quantity',
                        '% הנחה לשורה': 'discount_percentage',
                        'מחיר לאחר הנחה': 'price_after_discount',
                        'מחיר יחידה': 'price_per_unit',
                        'קוד איש מכירות': 'code_sales_person',
                        'תיאור הפריט/שירות': 'description_sku',
                        'מק"ט יצרן': 'code_manufacturer',
                        'סך הכול בשורה': 'total_in_line',
'מדינה/אזור יעד לייבוא': 'county_destination',
'מדינה/אזור מקור לייצוא': 'county_origin',
                        'סוג משלוח': 'delivery_type',})

print(df[['date', 'date_supply', 'sku', 'quantity', 'discount_percentage', 'price_after_discount', 'price_per_unit', 'code_sales_person', 'description_sku', 'code_manufacturer', 'total_in_line', 'county_destination', 'county_origin', 'delivery_type']])


df = df[['date', 'date_supply', 'sku', 'quantity', 'discount_percentage', 'price_after_discount', 'price_per_unit', 'code_sales_person', 'description_sku', 'code_manufacturer', 'total_in_line', 'delivery_type']]


df['price_after_discount'] = df['price_after_discount'].str.replace(',', '').str.replace('₪', '').astype(float)
df['quantity'] = df['quantity'].str.replace(',', '').str.replace('₪', '').astype(float)
df['price_per_unit'] = df['price_per_unit'].str.replace(',', '').str.replace('₪', '').astype(float)
df['total_in_line'] = df['total_in_line'].str.replace(',', '').str.replace('₪', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].str.replace(',', '').str.replace('%', '').astype(float)
print(df.info())
# plot the price_after_discount by date agg by sum month

df['price*quantity'] = df['price_after_discount'] * df['quantity']
# print num of nan in each column
df.to_csv('sap_clean_data.csv', index=False)
df.set_index('date', inplace=True)
df.resample('Y')['price*quantity'].sum().plot()


# save the data to a new csv file
