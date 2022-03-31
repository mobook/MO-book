#!/usr/bin/env python
# coding: utf-8

# # Download Historical Stock Data
# 
# Use this notebook to download daily trading data for a group of the stocks from Yahoo Finance. The trading data is stored in a designated sub-directory (default `./data/stocks/`) as individual `.csv` files for each stock. Subsequent notebooks can read and consolidate the stock price data. 
# 
# Run the cells in the notebook once to create data sets for use by other notebook, or to refresh a previously stored set of data. The function will overwrite any existing data sets.

# ## Installing and Testing `pandas_datareader`
# 
# The notebook uses the `pandas_datareader` module to read data from Yahoo Finance. Web interfaces for financial services are notoriously fickle and subject to change, and a particular issue with Google Colaboratory. The following cell tests if `pandas_datareader` installed and functional. It will attempt to upgrade and restart the Python kernel. If you encounter repeated errors please report this as an issue for this notebook. 

# In[3]:


import sys
import os

# attempt to install. If not found then try install
try:
    import pandas_datareader as pdr
    
except:
    get_ipython().system('pip install -q pandas_datareader')
    from pandas_datareader import data, wb, DataReader
    
# test download. If fail, then upgrade and restart kernal
try: 
    goog = pdr.DataReader("GOOG", "yahoo")
    print("pandas_datareader is installed and appears to be working correctly.")
except:
    get_ipython().system('pip install pandas_datareader --upgrade')
    os.kill(os.getpid(), 9)
    


# ## Stocks to Download
# 
# Edit the following cell to download a list of stock symbols from Yahoo Finance,  `n_years` to change the historical period, or change the data directory.

# In[1]:


import os

# list of stock symbols
assets = ['AXP', 'AAPL', 'AMGN', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',          'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',          'MSFT', 'NKE', 'PG','TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM']

# number of years
n_years = 3.0

# create data directory
data_dir = os.path.join('data', 'stocks')
os.makedirs(data_dir, exist_ok=True)


# ## Downloads
# 
# Run the following cell to download the historical stock data.

# In[2]:


import pandas as pd
import datetime as datetime

# historical period
end_date = datetime.datetime.today().date()
start_date = end_date - datetime.timedelta(round(n_years*365))

# get daily price data from yahoo finance
def get_stock_data(s, path=data_dir):
    try:
        print(f"Downloading {s:6s}", end="")
        data = pdr.DataReader(s, "yahoo", start_date, end_date)
        try:
            filename = os.path.join(data_dir, s + '.csv')
            data.to_csv(filename) 
            print(f" saved to {filename}")
        except: 
            print("save failed")
    except:
        print(f"download failed")      
    
for s in assets:
    get_stock_data(s)
    


# In[ ]:




