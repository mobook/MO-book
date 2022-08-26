#!/usr/bin/env python
# coding: utf-8

# # Markowitz Portfolio Optimization

# In[2]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# In[3]:


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as stats

import datetime as datetime

import sys
import os
import glob

import pyomo.environ as pyo


# In[ ]:




