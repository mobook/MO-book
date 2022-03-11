#!/usr/bin/env python
# coding: utf-8

# # Obtaining Historical Stock Data
# 
# Keywords: stock price data
# 
# The purpose of this notebook is to download historical trading data for a selected group of the stocks from Alpha Vantage for use with other notebooks. Use of this notebook requires you so enter your personal Alpha Vantage api key into a file `data/api_key.txt`.  The trading data is stored as individual `.csv` files in a designated directory. Subsequent notebooks read and consolidate that data into a singe file.  
# 
# You only need to run this notebook if you wish to analyze a different set of stocks, if you wish to update data for the existing set.

# ## Imports

# In[6]:


import os
import pandas as pd
import requests
import time
import datetime as datetime
from pandas_datareader import data, wb, DataReader


# ## Select Stocks to Download

# In[7]:


asset_symbols = ['AXP', 'AAPL', 'AMGN', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',                  'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',                  'MSFT', 'NKE', 'PG','TRV', 'UNH', 'V','VZ', 'WBA', 'WMT','XOM']

# historical period
end_date = datetime.datetime.today().date()
start_date = end_date - datetime.timedelta(3*365)


# In[7]:


# create data directory if none exists
data_dir = os.path.join('data', 'stocks')
os.makedirs(data_dir, exist_ok=True)

# get daily price data from yahoo financ
def get_stock_data(s, path=data_dir):
    print("downloading", s, end="")
    data = DataReader(s, "yahoo", start_date, end_date)
    filename = os.path.join(data_dir, s + '.csv')
    data.to_csv(filename)
    print(f" saved to {filename}")
    
for s in asset_symbols:
    get_stock_data(s)

