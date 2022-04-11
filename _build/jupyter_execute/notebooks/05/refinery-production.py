#!/usr/bin/env python
# coding: utf-8

# # Refinery Production and Shadow Pricing

# In[9]:


# MO-book helper code
get_ipython().system('curl -sO https://raw.githubusercontent.com/jckantor/MO-book/main/tools/mobook.py')
import mobook
mobook.svg()


# Inspired by Example 19.3 from Seborg, Edgar, Mellichamp, and Doyle. The changes include updating prices, solution using optimization modeling languages, and adding constraints to demonstrate the significance of duals and their interpretation as shadow prices.

# In[10]:


import cvxpy as cvxpy
import pandas as pd

products = pd.DataFrame({
    "gasoline": {"capacity": 24000, "price": 108},
    "kerosine": {"capacity":  2000, "price": 72},
    "fuel oil": {"capacity":  6000, "price": 63},
    "residual": {"capacity":  2500, "price": 30},
}).T

crudes = pd.DataFrame({
    "crude 1": {"available": 28000, "purchase cost": 72, "process cost": 1.5},
    "crude 2": {"available": 15000, "purchase cost": 45, "process cost": 3},
}).T

# note: volumetric yields may not add to 100%
yields = pd.DataFrame({
    "crude 1": {"gasoline": 80, "kerosine": 5, "fuel oil": 10, "residual": 5},
    "crude 2": {"gasoline": 44, "kerosine": 10, "fuel oil": 36, "residual": 10},
}).T

display(products)
display(crudes)
display(yields)


# ## Pyomo Model

# In[11]:


import pyomo.environ as pyo

m = pyo.ConcreteModel

m.CRUDES = pyo.Set(initialize=crudes.index)
m.PRODUCTS = pyo.Set(initialize=products.index)


# ## CVXPY Model
# 
# 

# In[12]:


import numpy as np
import cvxpy as cp

# decision variables
x = cp.Variable(len(crudes.index), pos=True, name="crudes")
y = cp.Variable(len(products.index), pos=True, name="products")

# objective
revenue = products["price"].to_numpy().T @ y
purchase_cost = crudes["purchase cost"].to_numpy().T @ x
process_cost = crudes["process cost"].to_numpy().T @ x
profit = revenue - purchase_cost - process_cost
objective = cp.Maximize(profit)

# constraints
balances = y == yields.to_numpy().T @ x/100
feeds = x <= crudes["available"].to_numpy()
capacity = y <= products["capacity"].to_numpy()
constraints = [balances, feeds, capacity]

# solution
problem = cp.Problem(objective, constraints)
problem.solve()


# ## Feeds

# In[13]:


results_crudes = crudes
results_crudes["consumption"] = x.value
results_crudes["shadow price"] = feeds.dual_value

display(results_crudes.round(1))


# ## Production

# In[14]:


results_products = products
results_products["production"] = y.value
results_products["unused capacity"] = products["capacity"] - y.value
results_products["shadow price"] = capacity.dual_value

display(results_products.round(1))


# ## Why are the shadow prices so high?
# 
# $$ y = A x $$

# In[15]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
ylim = 24000
xlim = 32000

ax.axvline(crudes["available"][0], linestyle='--', label="Crude 1")
ax.axhline(crudes["available"][1], linestyle='--', label="Crude 2")

xplot = np.linspace(0, xlim)
for product in products.index:
    b = 100*products.loc[product, "capacity"]/yields[product][1]
    m = - yields[product][0]/yields[product][1]
    line = ax.plot(xplot, m*xplot + b, label=product)
    ax.fill_between(xplot, m*xplot + b, 30000, color=line[0].get_color(), alpha=0.2)

ax.plot(x.value[0], x.value[1], 'ro', ms=10, label="max profit")
ax.set_title("Feasible operating regime")
ax.set_xlabel(crudes.index[0])
ax.set_ylabel(crudes.index[1])
ax.legend()
ax.set_xlim(0, xlim)
ax.set_ylim(0, ylim)


# ## Suggested Exercises
# 
# 1. Suppose the refinery makes a substantial investment to double kerosine production in order to increase profits. What becomes the limiting constraint?
# 
# 2. How do prices of crudes and products change the location of the optimum operating point?
# 
# 2. A refinery is a financial asset for the conversion of commodity crude oils into commodity hydrocarbons. What economic value can be assigned to owning the option to convert crude oils into other commodities?

# In[ ]:




