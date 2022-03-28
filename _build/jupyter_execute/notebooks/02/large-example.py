#!/usr/bin/env python
# coding: utf-8

# # Caroline's raw material planning
# 
# <img align='right' src='https://drive.google.com/uc?export=view&id=1FYTs46ptGHrOaUMEi5BzePH9Gl3YM_2C' width=200>
# 
# As we know, BIM produces logic and memory chips using copper, silicon, germanium and plastic. 
# 
# Each chip has the following consumption of materials:
# 
# | chip   | copper | silicon | germanium | plastic |
# |:-------|-------:|--------:|----------:|--------:|
# |Logic   |    0.4 |       1 |           |       1 |
# |Memory  |    0.2 |         |         1 |       1 |
# 
# BIM hired Caroline to manage the acquisition and the inventory of these raw materials. 
# 
# Caroline conducted a data analysis which lead to the following prediction of monthly demands for her trophies: 
# 
# | chip   | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
# |:-------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
# |Logic   |  88 | 125 | 260 | 217 | 238 | 286 | 248 | 238 | 265 | 293 | 259 | 244 |
# |Memory  |  47 |  62 |  81 |  65 |  95 | 118 |  86 |  89 |  82 |  82 |  84 | 66  |
# 
# As you recall, BIM has the following stock at the moment:
# 
# |copper|silicon|germanium|plastic|
# |-----:|------:|--------:|------:|
# |   480|  1000 |     1500|  1750 |
# 
# BIM would like to have at least the following stock at the end of the year:
# 
# |copper|silicon|germanium|plastic|
# |-----:|------:|--------:|------:|
# |   200|   500 |      500|  1000 |
# 
# Each product can be acquired at each month, but the unit prices vary as follows:
# 
# | product  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
# |:---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
# |copper    |   1 |   1 |   1 |   2 |   2 |   3 |   3 |   2 |   2 |   1 |   1 |   2 |
# |silicon   |   4 |   3 |   3 |   3 |   5 |   5 |   6 |   5 |   4 |   3 |   3 |   5 |
# |germanium |   5 |   5 |   5 |   3 |   3 |   3 |   3 |   2 |   3 |   4 |   5 |   6 |
# |plastic   | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
# 
# The inventory is limited by a capacity of a total of 9000 units per month, regardless of the composition of products in stock. 
# The holding costs of the inventory are 0.05 per unit per month regardless of the product.
# 
# Caroline cannot spend more than 5000 per month on acquisition.
# 
# Note that Caroline aims at minimizing the acquisition and holding costs of the materials while meeting the required quantities for production. 
# The production is made to order, meaning that no inventory of chips is kept.
# 
# Please help Caroline to model the material planning and solve it with the data above. 

# In[63]:


import sys
if 'google.colab' in sys.modules:
    import shutil
    if not shutil.which('pyomo'):
        get_ipython().system('pip install -q pyomo')
        assert(shutil.which('pyomo'))

    # cbc
    get_ipython().system('apt-get install -y -qq coinor-cbc')


# To be self contained... alternative is to upload and read a file. 

# In[64]:


demand_data = '''chip,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec
Logic,88,125,260,217,238,286,248,238,265,293,259,244
Memory,47,62,81,65,95,118,86,89,82,82,84,66'''


# In[65]:


from io import StringIO
import pandas as pd
demand_chips = pd.read_csv( StringIO(demand_data), index_col='chip' )
demand_chips


# In[66]:


price_data = '''product,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec
copper,1,1,1,2,2,3,3,2,2,1,1,2
silicon,4,3,3,3,5,5,6,5,4,3,3,5
germanium,5,5,5,3,3,3,3,2,3,4,5,6
plastic,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1'''


# In[67]:


price = pd.read_csv( StringIO(price_data), index_col='product' )
price


# # A possible resolution

# ## A simple dataframe with the consumptions

# In[68]:


use = dict()
use['Logic'] = { 'silicon' : 1, 'plastic' : 1, 'copper' : 4 }
use['Memory'] = { 'germanium' : 1, 'plastic' : 1, 'copper' : 2 }
use = pd.DataFrame.from_dict( use ).fillna(0).astype( int )
use


# ## A simple matrix multiplication

# In[69]:


demand = use.dot( demand_chips )
demand


# In[81]:


import pyomo.environ as pyo


# In[121]:


m = pyo.ConcreteModel()


# # Add the relevant data to the model

# In[122]:


m.Time        = demand.columns
m.Product     = demand.index
m.Demand      = demand
m.UnitPrice   = price
m.HoldingCost = .05
m.StockLimit  = 9000
m.Budget      = 2000


# In[123]:


m.existing = {'silicon' : 1000, 'germanium': 1500, 'plastic': 1750, 'copper' : 4800 }
m.desired  = {'silicon' :  500, 'germanium':  500, 'plastic': 1000, 'copper' : 2000 }


# # Some care to deal with the `time` index

# In[124]:


m.first = m.Time[0]
m.last  = m.Time[-1]
m.prev  = { j : i for i,j in zip(m.Time,m.Time[1:]) }


# # Variables for the decision (buy) and consequence (stock)

# In[125]:


m.buy   = pyo.Var( m.Product, m.Time, within=pyo.NonNegativeReals )
m.stock = pyo.Var( m.Product, m.Time, within=pyo.NonNegativeReals )


# # The constraints that balance acquisition with inventory and demand

# In[126]:


def BalanceRule( m, p, t ):
    if t == m.first:
        return m.existing[p] + m.buy[p,t] == m.Demand.loc[p,t] + m.stock[p,t]
    else:
        return m.buy[p,t] + m.stock[p,m.prev[t]] == m.Demand.loc[p,t] + m.stock[p,t]


# In[127]:


m.balance = pyo.Constraint( m.Product, m.Time, rule = BalanceRule )


# # The remaining constraints
# 
# Note that these rules are so simple, one liners, that it is better to just define them 'on the spot' as anonymous (or `'lambda`) functions. 

# ## Ensure the desired inventory at the end of the horizon

# In[128]:


m.finish = pyo.Constraint( m.Product, rule = lambda m, p : m.stock[p,m.last] >= m.desired[p] )


# ## Ensure that the inventory fits the capacity

# In[129]:


m.inventory = pyo.Constraint( m.Time, rule = lambda m, t : sum( m.stock[p,t] for p in m.Product ) <= m.StockLimit )


# ## Ensure that the acquisition fits the budget

# In[130]:


m.budget = pyo.Constraint( m.Time, rule = lambda m, t : sum( m.UnitPrice.loc[p,t]*m.buy[p,t] for p in m.Product ) <= m.Budget )


# In[131]:


m.obj = pyo.Objective( expr = sum( m.UnitPrice.loc[p,t]*m.buy[p,t] for p in m.Product for t in m.Time )
                              + sum( m.HoldingCost*m.stock[p,t] for p in m.Product for t in m.Time )
                      , sense = pyo.minimize )


# In[132]:


pyo.SolverFactory( 'gurobi_direct' ).solve(m)


# In[133]:


def ShowDouble( X, I,J ):
    return pd.DataFrame.from_records( [ [ X[i,j].value for j in J ] for i in I ], index=I, columns=J )


# In[134]:


ShowDouble( m.buy, m.Product, m.Time )


# In[118]:


ShowDouble( m.stock, m.Product, m.Time )


# In[119]:


ShowDouble( m.stock, m.Product, m.Time ).T.plot(drawstyle='steps-mid',grid=True, figsize=(20,4))


# # Notes
# 
# * The budget is not limitative. 
# * With the given budget the solution remains integer. 
# * Lowering the budget to 2000 forces acquiring fractional quantities. 
# * Lower values of the budget end up making the problem infeasible.
