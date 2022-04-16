#!/usr/bin/env python
# coding: utf-8

# # Minimum Risk-Free Rate of Return
# 
# 
# Application of Farka's lemma and convex duality. - Ar
# 
# 
# ## Farka's lemma
# 
# Farka's lemma, first published by the Hungarian mathematician and physicist Gyula Farkas in 1902, is one of foundations for the theory of linear programming and optimization. 
# 
# 
# ## Assets
# 
# * A: Price today is 30  Next year the price might go up to 40, stay at 30, or fall to 20
# * B: Price today is 40. Next year the price might go to up 50, stay at 40, or fall to 30.
# * I have the means to convert 1 unit of to 1 unit of B, A -> B, at a cost 5
# * A risk free return exists that converts 100 to 105.
# * I have $1000 to invest. 
# 
# $$P = P_A x_A + P_B x_B + P_f x_f$$

# In[ ]:


import pandas as pd

scenarios_a = [20, 30, 40]
scenarios_b = [30, 40, 50]



# ## Example
# 
# The owner of a house building business has decided to sell the operation and needs to establish a fair price. The business is highly variable due to the changing relationships between the cost of materials, labor, and the market price of existing houses. In years when the market price of houses falls below the cost of construction there is no reason to build houses for sale. The value of the operations is in those years when a profit can be earned that is hiring than a risk-free investment alternative.
# 
# | Scenario | Materials | Labor | Market Price | Operating Net |
# | :------- | :-------: | :---: | :----------: | :----: |
# | A | 100 | 300 | 350 | -10 |
# | B | 120 | 350 | 400 | 70 |
# | C | 80 | 250 | 350 | 20 |
# 
# * Cost of materials:  
# 
# of owning the business fo
# 
# * Convert 2A -> P (commodity goods). There's a cost to operate
# * Owner holds option to produce quantity q of P at time T
# * Multiple price scenarios for A and P
# * There's a risk free alternative investment with payoff Rf
# 
# Determine -- difference between seller and buyer

# In[ ]:




