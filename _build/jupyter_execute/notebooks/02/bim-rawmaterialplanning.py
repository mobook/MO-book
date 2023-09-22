#!/usr/bin/env python
# coding: utf-8

# ```{index} single: Pyomo; sets
# ```
# ```{index} single: Pyomo; decorators
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} single: solver; highs
# ```
# ```{index} single: application; production planning
# ```
# ```{index} single: application; demand forecasts
# ```
# ```{index} pandas dataframe
# ```
# 
# # BIM production using demand forecasts

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


# ## The problem: Optimal material acquisition and production planning using demand forecasts
# 
# This example is a continuation of the BIM chip production problem illustrated [here](bim.ipynb). Recall hat BIM produces logic and memory chips using copper, silicon, germanium, and plastic and that each chip requires the following quantities of raw materials:
# 
# | chip   | copper | silicon | germanium | plastic |
# |:-------|-------:|--------:|----------:|--------:|
# |logic   |    0.4 |       1 |         - |       1 |
# |memory  |    0.2 |       - |         1 |       1 |
# 
# BIM needs to carefully manage the acquisition and inventory of these raw materials based on the forecasted demand for the chips. Data analysis led to the following prediction of monthly demands:
# 
# | chip   | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
# |:-------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
# |logic   |  88 | 125 | 260 | 217 | 238 | 286 | 248 | 238 | 265 | 293 | 259 | 244 |
# |memory  |  47 |  62 |  81 |  65 |  95 | 118 |  86 |  89 |  82 |  82 |  84 | 66  |
# 
# At the beginning of the year, BIM has the following stock:
# 
# |copper|silicon|germanium|plastic|
# |-----:|------:|--------:|------:|
# |   480|  1000 |     1500|  1750 |
# 
# The company would like to have at least the following stock at the end of the year:
# 
# |copper|silicon|germanium|plastic|
# |-----:|------:|--------:|------:|
# |   200|   500 |      500|  1000 |
# 
# Each raw material can be acquired at each month, but the unit prices vary month by month as follows:
# 
# | product  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
# |:---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
# |copper    |   1 |   1 |   1 |   2 |   2 |   3 |   3 |   2 |   2 |   1 |   1 |   2 |
# |silicon   |   4 |   3 |   3 |   3 |   5 |   5 |   6 |   5 |   4 |   3 |   3 |   5 |
# |germanium |   5 |   5 |   5 |   3 |   3 |   3 |   3 |   2 |   3 |   4 |   5 |   6 |
# |plastic   | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
# 
# The inventory is limited by a capacity of a total of 9000 units per month, regardless of the type of material of products in stock. The holding costs of the inventory are 0.05 per unit per month regardless of the material type. Due to budget constraints, BIM cannot spend more than 5000 per month on acquisition.
# 
# BIM aims at minimizing the acquisition and holding costs of the materials while meeting the required quantities for production. The production is made to order, meaning that no inventory of chips is kept.
# 
# Let us model the material acquisition planning and solve it optimally based on the chip demand forecasted above. First import both the price and forecast chip demand as Pandas dataframes.

# In[2]:


import pandas as pd
import pyomo.environ as pyo
from io import StringIO
from IPython.display import display

demand_data = '''
chip, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
logic, 88, 125, 260, 217, 238, 286, 248, 238, 265, 293, 259, 244
memory, 47, 62, 81, 65, 95, 118, 86, 89, 82, 82, 84, 66
'''

price_data = '''
product, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
copper, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 2
silicon, 4, 3, 3, 3, 5, 5, 6, 5, 4, 3, 3, 5
germanium, 5, 5, 5, 3, 3, 3, 3, 2, 3, 4, 5, 6
plastic, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
'''

demand_chips = pd.read_csv(StringIO(demand_data), index_col='chip')
display(demand_chips)

price = pd.read_csv(StringIO(price_data), index_col='product')
display(price)


# We can also add a small dataframe with the consumptions and obtain the monthly demand for each raw material using a simple matrix multiplication.

# In[3]:


use = dict()
use['logic'] = {'silicon': 1, 'plastic': 1, 'copper': 4}
use['memory'] = {'germanium': 1, 'plastic': 1, 'copper': 2}
use = pd.DataFrame.from_dict(use).fillna(0).astype(int)
display(use)

demand = use.dot(demand_chips)
display(demand)


# ## The optimization model
# 
# Define the set of raw material $P=\{\text{copper},\text{silicon},\text{germanium},\text{plastic}\}$ and $T$ the set of the $12$ months of the year. Let
# 
# - $x_{pt} \geq 0$ be the variable describing the amount of raw material $p \in P$ acquired in month $t \in T$;
# 
# - $s_{pt} \geq 0$ be the variable describing the amount of raw material $p \in P$ left in stock at the end of month $t \in T$. Note that these values are uniquely determined by the $x$ variables, but we keep these additional variables to ease the modeling. 
# 
# The total cost is the objective function of our optimal acquisition and production problem. If $\pi_{pt}$ is the unit price of product $p \in P$ in month $t \in T$ and $h_{pt}$ the unit holding costs (which happen to be constant) we can express the total cost as:
# 
# $$
#     \sum_{p\in P}\sum_{t \in T}\pi_{pt}x_{pt} + \sum_{p\in P}\sum_{t \in T} h_{pt} s_{pt}.
# $$
# 
# Let us now focus on the constraints. If $\beta \geq 0$ denotes the monthly acquisition budget, the budget constraint can be expressed as:
# 
# $$
#     \sum_{p\in P} \pi_{pt}x_{pt} \leq \beta \quad \forall t \in T.
# $$
# 
# Further, we constrain the inventory to be always the storage capacity $\ell \geq 0$ using:
# 
# $$
#     \sum_{p\in P} s_{pt} \leq \ell \quad \forall t \in T.
# $$
# 
# Next, we add another constraint to fix the value of the variables $s_{pt}$ by balancing the acquired amounts with the previous inventory and the demand $\delta_{pt}$ which for each month is implied by the total demand for the chips of both types. Note that $t-1$ is defined as the initial stock when $t$ is the first period, that is \texttt{January}. This can be obtained with additional variables $s$ made equal to those values or with a rule that specializes, as in the code below.  
# 
# $$
#     x_{pt} + s_{p,t-1} = \delta_{pt} + s_{pt} \quad \forall p \in P, t \in T.
# $$
# 
# Finally, we capture the required minimum inventory levels in December with the constraint.
# 
# $$
#     s_{p \textrm{Dec}} \geq \Omega_p \quad \forall p \in P,
# $$
# 
# where $(\Omega_p)_{p \in P}$ is the vector with the desired end inventories.
# 
# Here is the Pyomo implementation of this LP.

# In[4]:


def BIMProductAcquisitionAndInventory(demand, acquisition_price, existing, desired, stock_limit, month_budget):
    
    m = pyo.ConcreteModel('BIM product acquisition and inventory')
    
    periods = demand.columns
    products = demand.index 
    first = periods[0] 
    prev = {j : i for i,j in zip(periods,periods[1:])}
    last = periods[-1]
    
    m.T = pyo.Set(initialize=periods)
    m.P = pyo.Set(initialize=products)
    m.PT = m.P * m.T # to avoid internal set bloat
    
    m.x = pyo.Var(m.PT, domain=pyo.NonNegativeReals)
    m.s = pyo.Var(m.PT, domain=pyo.NonNegativeReals)
    
    @m.Param(m.PT)
    def pi(m, p, t):
        return acquisition_price.loc[p][t]
    
    @m.Param(m.PT)
    def h(m, p, t): 
        return .05 # the holding cost
    
    @m.Param(m.PT)
    def delta(m, p, t):
        return demand.loc[p,t]
    
    @m.Expression()
    def acquisition_cost(m):
        return pyo.quicksum(m.pi[p,t] * m.x[p,t] for p in m.P for t in m.T)
    
    @m.Expression()
    def inventory_cost(m):
        return pyo.quicksum(m.h[p,t] * m.s[p,t] for p in m.P for t in m.T)
    
    @m.Objective(sense=pyo.minimize)
    def total_cost(m):
        return m.acquisition_cost + m.inventory_cost
    
    @m.Constraint(m.PT)
    def balance(m, p, t):
        if t == first:
            return existing[p] + m.x[p,t] == m.delta[p,t] + m.s[p,t]
        return m.x[p,t] + m.s[p,prev[t]] == m.delta[p,t] + m.s[p,t]
        
    @m.Constraint(m.P)
    def finish(m, p):
        return m.s[p,last] >= desired[p]
    
    @m.Constraint(m.T)
    def inventory(m, t):
        return pyo.quicksum(m.s[p,t] for p in m.P ) <= stock_limit
    
    @m.Constraint(m.T)
    def budget(m, t):
        return pyo.quicksum(m.pi[p,t] * m.x[p,t] for p in m.P ) <= month_budget
    
    return m


# We now can create an instance of the model using the provided data and solve it.

# In[5]:


import matplotlib.pyplot as plt
import numpy as np

budget = 5000
m = BIMProductAcquisitionAndInventory(demand, price, 
           {'silicon' : 1000, 'germanium': 1500, 'plastic': 1750, 'copper': 4800}, 
           {'silicon' :  500, 'germanium':  500, 'plastic': 1000, 'copper': 2000},
           9000, budget)

SOLVER.solve(m)

def show_table_of_pyomo_variables(X, I, J):
    return pd.DataFrame.from_records([[pyo.value(X[i,j]) for j in J] for i in I], index=I, columns=J).round(2)

def report_pyomo_solution(m):
    print('\nThe optimal amounts of raw materials to acquire in each month are:')
    display(show_table_of_pyomo_variables(m.x, m.P, m.T))

    print('\nThe corresponding optimal stock levels in each months are:')
    stock = show_table_of_pyomo_variables(m.s, m.P, m.T)
    display(stock)

    print('\nThe stock levels can be visualized as follows')
    plt.rcParams['font.size'] = 14
    colors = plt.get_cmap('tab20c')
    equidistant_colors = [colors(0.0), colors(0.2), colors(0.6), colors(0.4)]
    ax = stock.T.plot(drawstyle='steps-mid', lw=2, grid=True, figsize=(10, 5), color=equidistant_colors)
    ax.legend(loc='upper right')
    ax.set_xticks(ticks=range(len(stock.columns)))
    ax.set_xticklabels(stock.columns)
    ax.set_xlabel('Month')
    ax.set_ylabel('Stock level')
    plt.tight_layout()
    plt.show()
    
report_pyomo_solution(m)


# Here is a different solution corresponding to the situation where the budget is much lower, namely 2000.

# In[6]:


budget = 2000
m = BIMProductAcquisitionAndInventory( demand, price, 
           {'silicon': 1000, 'germanium': 1500, 'plastic': 1750, 'copper': 4800}, 
           {'silicon': 500, 'germanium': 500, 'plastic': 1000, 'copper': 2000},
           9000, budget)

SOLVER.solve(m)

report_pyomo_solution(m)


# Looking at the two optimal solutions corresponding to different budgets, we can note that:
# * The budget is not limitative;
# * With the initial budget of 5000 the solution remains integer;
# * Lowering the budget to 2000 forces acquiring fractional quantities;
# * Lower values of the budget end up making the problem infeasible.

# ### A more parsimonious model
# 
# We can create a more parsimonious model with fewer variabels by getting rid of the auxiliary variables $s_{pt}$. Here is the corresponding implementation in Pyomo:

# In[7]:


def BIMProductAcquisitionAndInventory_v2(demand, acquisition_price, existing, desired, stock_limit, month_budget):
    
    m = pyo.ConcreteModel('Product acquisition and inventory')
    
    periods = demand.columns
    products = demand.index 
    first = periods[0] 
    prev = {j : i for i,j in zip(periods,periods[1:])}
    last = periods[-1]
    
    m.T = pyo.Set(initialize=periods)
    m.P = pyo.Set(initialize=products)
    m.PT = m.P * m.T # to avoid internal set bloat
    
    m.x = pyo.Var(m.PT, domain=pyo.NonNegativeReals)
    
    @m.Param(m.PT)
    def pi(m, p, t):
        return acquisition_price.loc[p][t]
    
    @m.Param(m.PT)
    def h(m, p, t): 
        return .05 # the holding cost
    
    @m.Param(m.PT)
    def delta(m, p, t):
        return demand.loc[p,t]
    
    @m.Expression(m.PT)
    def s(m, p, t):
        if t == first:
            return existing[p] + m.x[p,t] - m.delta[p,t]
        return m.x[p,t] + m.s[p,prev[t]] - m.delta[p,t]
        
    @m.Constraint(m.PT)
    def non_negative_stock(m, p, t):
        return m.s[p,t] >= 0
    
    @m.Expression()
    def acquisition_cost(m):
        return pyo.quicksum(m.pi[p,t] * m.x[p,t] for p in m.P for t in m.T)
    
    @m.Expression()
    def inventory_cost(m):
        return pyo.quicksum(m.h[p,t] * m.s[p,t] for p in m.P for t in m.T)
    
    @m.Objective(sense=pyo.minimize)
    def total_cost(m):
        return m.acquisition_cost + m.inventory_cost
            
    @m.Constraint(m.P)
    def finish(m, p):
        return m.s[p,last] >= desired[p]
    
    @m.Constraint(m.T)
    def inventory(m, t):
        return pyo.quicksum(m.s[p,t] for p in m.P) <= stock_limit
    
    @m.Constraint(m.T)
    def budget(m, t):
        return pyo.quicksum(m.pi[p,t] * m.x[p,t] for p in m.P) <= month_budget
    
    return m


# In[8]:


m = BIMProductAcquisitionAndInventory_v2( demand, price, 
           {'silicon' : 1000, 'germanium': 1500, 'plastic': 1750, 'copper' : 4800 }, 
           {'silicon' :  500, 'germanium':  500, 'plastic': 1000, 'copper' : 2000 },
           9000, 2000 )

SOLVER.solve(m)

report_pyomo_solution(m)

