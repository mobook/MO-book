#!/usr/bin/env python
# coding: utf-8

# ```{index} single: Pyomo; block
# ```
# ```{index} single: Pyomo; sets
# ```
# ```{index} single: Pyomo; parameters
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} single: application; production planning
# ```
# ```{index} pandas dataframe
# ```
# 
# # BIM production revisited

# ## Preamble: Install Pyomo and a solver
# 
# This cell selects and verifies a global SOLVER for the notebook.
# 
# If run on Google Colab, the cell installs Pyomo and HiGHS, then sets SOLVER to 
# use the Highs solver via the appsi module. If run elsewhere, it assumes Pyomo and CBC
# have been previously installed and sets SOLVER to use the CBC solver via the Pyomo 
# SolverFactory. It then verifies that SOLVER is available.

# In[1]:


import sys

if 'google.colab' in sys.modules:
    get_ipython().system('pip install pyomo >/dev/null 2>/dev/null')
    get_ipython().system('pip install highspy >/dev/null 2>/dev/null')

    from pyomo.contrib import appsi
    SOLVER = appsi.solvers.Highs(only_child_vars=False)
    
else:
    from pyomo.environ import SolverFactory
    SOLVER = SolverFactory('cbc')

assert SOLVER.available(), f"Solver {SOLVER} is not available."


# ## Problem description
# 
# We consider BIM raw material planning, but now with more sophisticated pricing and acquisition protocols. There are now three suppliers, each of which can deliver the following materials:
#  - A: **silicon**, **germanium** and **plastic**
#  - B: **copper**
#  - C: all of the above
#  
# For the suppliers, the following conditions apply. Copper should be acquired in multiples of 100 gram, since it is delivered in sheets of 100 gram. Unitary materials such as silicon, germanium and plastic may be acquired in any number, but the price is in batches of 100. Meaning that 30 units of silicon with 10 units of germanium and 50 units of plastic cost as much as 1 unit of silicon but half as much as 30 units of silicon with 30 units of germanium and 50 units of plastic. Furthermore, supplier C sells all materials and offers a discount if purchased together: 100 gram of copper and a batch of unitary material cost just 7. This set price is only applied to pairs, meaning that 100 gram of copper and 2 batches cost 13.
# 
# The summary of the prices in &euro; is given in the following table:
# 
# |Supplier|Copper per sheet of 100 gram|Batch of units|Together|
# |:-------|---------------------:|-----------------:|-------:|
# | A      |                    - |                5 |      - |
# | B      |                    3 |                - |      - |
# | C      |                    4 |                6 |      7 |
# 
# Next, for stocked products inventory costs are incurred, whose summary is given in the following table:
# 
# |Copper per 10 gram| Silicon per unit| Germanium per unit|Plastic per unit|
# |---:|-------:|---:|-----:|
# | 0.1|   0.02 |0.02| 0.02 |
# 
# The holding price of copper is per 10 gram and the copper stocked is rounded up to multiples of 10 grams, meaning that 12 grams pay for 20. 
# 
# The capacity limitations of the warehouse allow for a maximum of $10$ kilogram of copper in stock at any moment, but there are no practical limitations to the number of units of unitary products in stock.
# 
# Recall that BIM has the following stock at the beginning of the year:
# 
# |Copper |Silicon |Germanium |Plastic|
# |---:|-------:|---:|-----:|
# | 480|   1000 |1500| 1750 |
# 
# The company would like to have at least the following stock at the end of the year:
# 
# |Copper |Silicon |Germanium |Plastic|
# |---:|-------:|---:|-----:|
# | 200|    500 | 500| 1000 |
# 
# The goal is to build an optimization model using the data above and solve it to minimize the acquisition and holding costs of the products while meeting the required quantities for production. The production is made-to-order, meaning that no inventory of chips is kept.
# 

# In[2]:


from io import StringIO
import pandas as pd

demand_data = '''
chip, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
logic, 88, 125, 260, 217, 238, 286, 248, 238, 265, 293, 259, 244
memory, 47, 62, 81, 65, 95, 118, 86, 89, 82, 82, 84, 66
'''

demand_chips = pd.read_csv(StringIO(demand_data), index_col='chip' )
display(demand_chips)


# In[3]:


use = dict()
use['logic'] = {'silicon' : 1, 'plastic' : 1, 'copper' : 4}
use['memory'] = {'germanium' : 1, 'plastic' : 1, 'copper' : 2}
use = pd.DataFrame.from_dict(use).fillna(0).astype(int)
display(use)


# In[4]:


demand = use.dot(demand_chips)
display(demand)


# In[5]:


def Table1d(m, J, retriever ):
    return pd.DataFrame([ 0+retriever(m,j) for j in J ], index=J ).T

def Table2d(m, I, J, retriever ):
    return pd.DataFrame.from_records([ [ 0+retriever(m, i, j ) for j in J ] for i in I ], index=I, columns=J )

def Table3d(m, I, J, names, K, retriever ):
    index = pd.MultiIndex.from_product([I,J], names=names )
    return pd.DataFrame.from_records([ [ 0+retriever(m,i,j,k) for k in K ] for i in I for j in J ],index=index,columns=K )


# In[6]:


import pyomo.environ as pyo

def BIMproduction_v1(demand, existing, desired, stock_limit,
                supplying_copper, supplying_batches, 
                price_copper_sheet, price_batch, discounted_price, 
                batch_size, copper_sheet_mass, copper_bucket_size,
                unitary_products, unitary_holding_costs
                ):
    
    m = pyo.ConcreteModel('Product acquisition and inventory with sophisticated prices')
    
    periods = demand.columns
    products = demand.index 
    first = periods[0] 
    prev = {j : i for i,j in zip(periods,periods[1:])}
    last = periods[-1]
    
    @m.Param(products, periods)
    def delta(m, p, t):
        return demand.loc[p,t]
    
    @m.Param(supplying_batches)
    def pi(m, s):
        return price_batch[s]
    
    @m.Param(supplying_copper)
    def kappa(m, s):
        return price_copper_sheet[s]
    
    @m.Param()
    def beta(m):
        return price_batch['C']+price_copper_sheet['C']-discounted_price
    
    @m.Param(products)
    def gamma(m, p):
        return unitary_holding_costs[p]
    
    @m.Param(products)
    def Alpha(m, p):
        return existing[p]
    
    @m.Param(products)
    def Omega(m, p):
        return desired[p]
    
    m.y = pyo.Var(periods, supplying_copper, within=pyo.NonNegativeIntegers )
    m.x = pyo.Var(unitary_products, periods, supplying_batches, within=pyo.NonNegativeReals )
    m.s = pyo.Var(products, periods, within=pyo.NonNegativeReals )
    m.u = pyo.Var(products, periods, within=pyo.NonNegativeReals )
    m.b = pyo.Var(periods, supplying_batches, within=pyo.NonNegativeIntegers )
    m.p = pyo.Var(periods, within=pyo.NonNegativeIntegers )
    m.r = pyo.Var(periods, within=pyo.NonNegativeIntegers )

    @m.Constraint(periods, supplying_batches )
    def units_in_batches(m, t, s ):
        return pyo.quicksum(m.x[p,t,s] for p in unitary_products ) <= batch_size*m.b[t,s]
    
    @m.Constraint(periods )
    def copper_in_buckets(m, t ):
        return m.s['copper',t] <= copper_bucket_size*m.r[t]
    
    @m.Constraint(periods )
    def inventory_capacity(m, t ):
        return m.s['copper',t] <= stock_limit

    @m.Constraint(periods )
    def pairs_in_batches(m, t ):
        return m.p[t] <= m.b[t,'C']

    @m.Constraint(periods )
    def pairs_in_sheets(m, t ):
        return m.p[t] <= m.y[t,'C']
    
    @m.Constraint(periods, products )
    def bought(m, t, p ):
        if p == 'copper':
            return m.u[p,t] == copper_sheet_mass*pyo.quicksum(m.y[t,s] for s in supplying_copper )
        else:
            return m.u[p,t] == pyo.quicksum(m.x[p,t,s] for s in supplying_batches )
            
    @m.Expression()
    def acquisition_cost(m ):
        return pyo.quicksum(
                    pyo.quicksum(m.pi[s]*m.b[t,s] for s in supplying_batches ) \
                  + pyo.quicksum(m.kappa[s]*m.y[t,s] for s in supplying_copper ) \
                  - m.beta * m.p[t] for t in periods )   
    
    @m.Expression()
    def inventory_cost(m ):
        return pyo.quicksum(m.gamma['copper']*m.r[t] + \
                             pyo.quicksum(m.gamma[p]*m.s[p,t] for p in unitary_products ) \
                            for t in periods )
    
    @m.Objective(sense=pyo.minimize )
    def total_cost(m ):
        return m.acquisition_cost + m.inventory_cost
    
    @m.Constraint(products, periods )
    def balance(m, p, t ):
        if t == first:
            return m.Alpha[p] + m.u[p,t] == m.delta[p,t] + m.s[p,t]
        else:
            return m.u[p,t] + m.s[p,prev[t]] == m.delta[p,t] + m.s[p,t]
        
    @m.Constraint(products )
    def finish(m, p ):
        return m.s[p,last] >= m.Omega[p]
    
    return m


# In[7]:


m1 = BIMproduction_v1(demand = demand, 
    existing = {'silicon' : 1000, 'germanium': 1500, 'plastic': 1750, 'copper' : 4800 }, 
    desired = {'silicon' :  500, 'germanium':  500, 'plastic': 1000, 'copper' : 2000 }, 
    stock_limit = 10000,
    supplying_copper = [ 'B', 'C' ],
    supplying_batches = [ 'A', 'C' ],
    price_copper_sheet = { 'B': 300, 'C': 400 }, 
    price_batch = { 'A': 500, 'C': 600 }, 
    discounted_price = 700, 
    batch_size = 100,
    copper_sheet_mass = 100,
    copper_bucket_size = 10,
    unitary_products = [ 'silicon', 'germanium', 'plastic' ], 
    unitary_holding_costs = { 'copper': 10, 'silicon' : 2, 'germanium': 2, 'plastic': 2 }
    )

SOLVER.solve(m1).write()


# In[8]:


stock = Table2d(m1, demand.index, demand.columns, lambda m,i,j : pyo.value(m.s[i,j]) )
display(stock)


# In[9]:


import matplotlib.pyplot as plt, numpy as np

stock.T.plot(drawstyle='steps-mid',grid=True, figsize=(13,4))
plt.xticks(np.arange(len(stock.columns)),stock.columns)
plt.show()


# In[10]:


Table1d(m1, J = demand.columns, retriever = lambda m, j : pyo.value(m.p[j] ) )


# In[11]:


Table2d(m1, demand.index, demand.columns, lambda m,i,j : pyo.value(m.u[i,j]) )


# In[12]:


Table2d(m1, [ 'A', 'C' ], demand.columns, lambda m,i,j : pyo.value(m.b[j,i]) )


# In[13]:


Table2d(m1, [ 'B', 'C' ], demand.columns, lambda m,i,j : pyo.value(m.y[j,i]) )


# In[14]:


x = Table3d(m1, 
        I        = [ 'A', 'C' ], 
        J        = [ 'silicon', 'germanium', 'plastic' ], 
        names    = ['supplier','materials'], 
        K        = demand.columns, 
        retriever= lambda m,i,j,k : 0+pyo.value(m.x[j,k,i] ) 
        )
x


# In[15]:


def BIMproduction_v2(demand, existing, desired, 
                stock_limit,
                supplying_copper, supplying_batches, price_copper_sheet, price_batch, discounted_price, 
                batch_size, copper_sheet_mass, copper_bucket_size,
                unitary_products, unitary_holding_costs
                ):
    m = pyo.ConcreteModel('Product acquisition and inventory with sophisticated prices in blocks' )
    
    periods  = demand.columns
    products = demand.index 
    first    = periods[0] 
    prev     = { j : i for i,j in zip(periods,periods[1:]) }
    last     = periods[-1]
    
    m.T = pyo.Set(initialize=periods )
    m.P = pyo.Set(initialize=products )
    
    m.PT = m.P * m.T # to avoid internal set bloat
    
    @m.Block(m.T )
    def A(b):
        b.x = pyo.Var(supplying_batches, products, within=pyo.NonNegativeReals )
        b.b = pyo.Var(supplying_batches, within=pyo.NonNegativeIntegers )
        b.y = pyo.Var(supplying_copper, within=pyo.NonNegativeIntegers )
        b.p = pyo.Var(within=pyo.NonNegativeIntegers )

        @b.Constraint(supplying_batches)
        def in_batches(b, s):
            return pyo.quicksum(b.x[s,p] for p in products ) <= batch_size*b.b[s]

        @b.Constraint()
        def pairs_in_batches(b):
            return b.p <= b.b['C']

        @b.Constraint()
        def pairs_in_sheets(b):
            return b.p <= b.y['C']
        
        @b.Expression(products)
        def u(b, p):
            if p == 'copper':
                return copper_sheet_mass*pyo.quicksum(b.y[s] for s in supplying_copper )
            return pyo.quicksum(b.x[s,p] for s in supplying_batches )
            
        @b.Expression()
        def cost(b):
            discount = price_batch['C']+price_copper_sheet['C']-discounted_price
            return pyo.quicksum(price_copper_sheet[s]*b.y[s] for s in supplying_copper ) \
                + pyo.quicksum(price_batch[s]*b.b[s] for s in supplying_batches ) \
                - discount * b.p    
    
    @m.Block(m.T )
    def I(b ):
        b.s = pyo.Var(products, within=pyo.NonNegativeReals )
        b.r = pyo.Var(within=pyo.NonNegativeIntegers )
        
        @b.Constraint()
        def copper_in_buckets(b):
            return b.s['copper'] <= copper_bucket_size*b.r
        
        @b.Constraint()
        def capacity(b ):
            return b.s['copper'] <= stock_limit

        @b.Expression()
        def cost(b ):
            return unitary_holding_costs['copper']*b.r + \
                pyo.quicksum(unitary_holding_costs[p]*b.s[p] for p in unitary_products )
            
    @m.Param(m.PT )
    def delta(m,t,p):
        return demand.loc[t,p]
    
    @m.Expression()
    def acquisition_cost(m ):
        return pyo.quicksum(m.A[t].cost for t in m.T )
    
    @m.Expression()
    def inventory_cost(m ):
        return pyo.quicksum(m.I[t].cost for t in m.T )
    
    @m.Objective(sense=pyo.minimize )
    def total_cost(m ):
        return m.acquisition_cost + m.inventory_cost
    
    @m.Constraint(m.PT )
    def balance(m, p, t ):
        if t == first:
            return existing[p] + m.A[t].u[p] == m.delta[p,t] + m.I[t].s[p]
        else:
            return m.A[t].u[p] + m.I[prev[t]].s[p] == m.delta[p,t] + m.I[t].s[p]
        
    @m.Constraint(m.P )
    def finish(m, p ):
        return m.I[last].s[p] >= desired[p]
    
    return m


# In[16]:


m2 = BIMproduction_v2(demand = demand, 
    existing = {'silicon' : 1000, 'germanium': 1500, 'plastic': 1750, 'copper' : 4800 }, 
    desired = {'silicon' :  500, 'germanium':  500, 'plastic': 1000, 'copper' : 2000 }, 
    stock_limit = 10000,
    supplying_copper = [ 'B', 'C' ],
    supplying_batches = [ 'A', 'C' ],
    price_copper_sheet = { 'B': 300, 'C': 400 }, 
    price_batch = { 'A': 500, 'C': 600 }, 
    discounted_price = 700, 
    batch_size = 100,
    copper_sheet_mass = 100,
    copper_bucket_size = 10,
    unitary_products = [ 'silicon', 'germanium', 'plastic' ], 
    unitary_holding_costs = { 'copper': 10, 'silicon' : 2, 'germanium': 2, 'plastic': 2 }
    )

SOLVER.solve(m2).write()


# In[17]:


Table3d(m2, 
        I        = [ 'A', 'C' ], 
        J        = [ 'silicon', 'germanium', 'plastic' ], 
        names    = ['supplier','materials'], 
        K        = m2.T, 
        retriever= lambda m,i,j,k : 0+pyo.value(m.A[k].x[i,j] ) 
        )


# In[18]:


def BIMproduction_v3( demand, existing, desired, 
                stock_limit,
                supplying_copper, supplying_batches, price_copper_sheet, price_batch, discounted_price, 
                batch_size, copper_sheet_mass, copper_bucket_size,
                unitary_products, unitary_holding_costs
                ):
    m = pyo.ConcreteModel( 'Product management with sophisticated prices, blocks and redundant variables' )
    
    periods  = demand.columns
    products = demand.index 
    first    = periods[0] 
    prev     = { j : i for i,j in zip(periods,periods[1:]) }
    last     = periods[-1]
    
    m.T = pyo.Set( initialize=periods )
    m.P = pyo.Set( initialize=products )
    
    m.PT = m.P * m.T # to avoid internal set bloat
    
    m.x = pyo.Var( m.PT, within=pyo.NonNegativeReals )
    
    @m.Block( m.T )
    def A( b ):
        b.x = pyo.Var( supplying_batches, products, within=pyo.NonNegativeReals )
        b.b = pyo.Var( supplying_batches, within=pyo.NonNegativeIntegers )
        b.y = pyo.Var( supplying_copper, within=pyo.NonNegativeIntegers )
        b.p = pyo.Var( within=pyo.NonNegativeIntegers )

        @b.Constraint( supplying_batches )
        def in_batches( b, s ):
            return pyo.quicksum( b.x[s,p] for p in products ) <= batch_size*b.b[s]

        @b.Constraint()
        def pairs_in_batches( b ):
            return b.p <= b.b['C']

        @b.Constraint()
        def pairs_in_sheets( b ):
            return b.p <= b.y['C']
        
        @b.Expression( products )
        def u( b, p ):
            if p == 'copper':
                return copper_sheet_mass*pyo.quicksum( b.y[s] for s in supplying_copper )
            return pyo.quicksum( b.x[s,p] for s in supplying_batches )
            
        @b.Expression()
        def cost( b ):
            discount = price_batch['C']+price_copper_sheet['C']-discounted_price
            return pyo.quicksum( price_copper_sheet[s]*b.y[s] for s in supplying_copper ) \
                + pyo.quicksum( price_batch[s]*b.b[s] for s in supplying_batches ) \
                - discount * b.p    
    
    @m.Block( m.T )
    def I( b ):
        b.s = pyo.Var( products, within=pyo.NonNegativeReals )
        b.r = pyo.Var( within=pyo.NonNegativeIntegers )
        
        @b.Constraint()
        def copper_in_buckets(b):
            return b.s['copper'] <= copper_bucket_size*b.r
        
        @b.Constraint()
        def capacity( b ):
            return b.s['copper'] <= stock_limit

        @b.Expression()
        def cost( b ):
            return unitary_holding_costs['copper']*b.r + \
                pyo.quicksum( unitary_holding_costs[p]*b.s[p] for p in unitary_products )
            
    @m.Param( m.PT )
    def delta(m,t,p):
        return demand.loc[t,p]
    
    @m.Expression()
    def acquisition_cost( m ):
        return pyo.quicksum( m.A[t].cost for t in m.T )
    
    @m.Expression()
    def inventory_cost( m ):
        return pyo.quicksum( m.I[t].cost for t in m.T )
    
    @m.Objective( sense=pyo.minimize )
    def total_cost( m ):
        return m.acquisition_cost + m.inventory_cost
    
    @m.Constraint( m.PT )
    def match( m, p, t ):
        return m.x[p,t] == m.A[t].u[p]
       
    @m.Constraint( m.PT )
    def balance( m, p, t ):
        if t == first:
            return existing[p] + m.x[p,t] == m.delta[p,t] + m.I[t].s[p]
        else:
            return m.x[p,t] + m.I[prev[t]].s[p] == m.delta[p,t] + m.I[t].s[p]
        
    @m.Constraint( m.P )
    def finish( m, p ):
        return m.I[last].s[p] >= desired[p]
    
    return m


# In[19]:


m3 = BIMproduction_v3( demand = demand, 
    existing = {'silicon' : 1000, 'germanium': 1500, 'plastic': 1750, 'copper' : 4800 }, 
    desired = {'silicon' :  500, 'germanium':  500, 'plastic': 1000, 'copper' : 2000 }, 
    stock_limit = 10000,
    supplying_copper = [ 'B', 'C' ],
    supplying_batches = [ 'A', 'C' ],
    price_copper_sheet = { 'B': 300, 'C': 400 }, 
    price_batch = { 'A': 500, 'C': 600 }, 
    discounted_price = 700, 
    batch_size = 100,
    copper_sheet_mass = 100,
    copper_bucket_size = 10,
    unitary_products = [ 'silicon', 'germanium', 'plastic' ], 
    unitary_holding_costs = { 'copper': 10, 'silicon' : 2, 'germanium': 2, 'plastic': 2 }
    )

SOLVER.solve(m3).write()


# In[20]:


Table1d(m3, J = m3.T, retriever = lambda m, j : pyo.value( m.A[j].p ) )


# In[21]:


Table2d(m3, I=m3.P, J=m3.T, retriever = lambda m, i, j : pyo.value( 0+m.x[i,j] ) )


# In[22]:


Table2d(m3, I=['B','C'], J=m3.T, retriever = lambda m, i, j : pyo.value( 0+m.A[j].y[i] ) )


# In[23]:


Table2d(m3, I=m3.P, J=m3.T, retriever = lambda m, i, j : pyo.value( m.I[j].s[i] ) )


# In[24]:


Table2d(m3, I=['A','C'], J=m3.T, retriever=lambda m, i, j : pyo.value( m.A[j].b[i] ) )


# In[25]:


Table3d(m3, 
    I = [ 'A', 'C'], 
    J = [ 'silicon', 'germanium', 'plastic'], 
    names = ['supplier', 'materials'], 
    K = m3.T, 
    retriever = lambda m, i, j, k : 0 + pyo.value(m.A[k].x[i,j]) 
)

