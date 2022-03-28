#!/usr/bin/env python
# coding: utf-8

# # Download Historical Stock Data
# 
# Use this notebook to download daily trading data for a selected group of the stocks from Yahoo Finance. The trading data is stored as individual `.csv` files in a designated directory. Subsequent notebooks read and consolidate that data into a singe file.  
# 
# You only need to run this notebook to analyze a different set of stocks or to update data for the existing set. The function will overwrite any existing data sets.

# ## Stocks to Download

# In[1]:


import os

# list of stock symbols
asset_symbols = ['AXP', 'AAPL', 'AMGN', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',                  'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',                  'MSFT', 'NKE', 'PG','TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM']

# number of years
n_years = 3.0

# create data directory
data_dir = os.path.join('data', 'stocks')
os.makedirs(data_dir, exist_ok=True)


# ## Downloads

# In[2]:


import os
import pandas as pd
import datetime as datetime
from pandas_datareader import data, wb, DataReader

# historical period
end_date = datetime.datetime.today().date()
start_date = end_date - datetime.timedelta(round(n_years*365))

# get daily price data from yahoo finance
def get_stock_data(s, path=data_dir):
    try:
        print(f"Downloading {s:6s}", end="")
        data = DataReader(s, "yahoo", start_date, end_date)
        try:
            filename = os.path.join(data_dir, s + '.csv')
            data.to_csv(filename) 
            print(f" saved to {filename}")
        except: 
            print("save failed")
    except:
        print(f"download failed")      
    
for s in asset_symbols:
    get_stock_data(s)


# In[ ]:




