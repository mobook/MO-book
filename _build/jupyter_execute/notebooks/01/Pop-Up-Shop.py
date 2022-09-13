#!/usr/bin/env python
# coding: utf-8

# # Scenario Analysis: Pop Up Shop
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Weich_Couture_Alpaca%2C_D%C3%BCsseldorf%2C_December_2020_%2809%29.jpg/300px-Weich_Couture_Alpaca%2C_D%C3%BCsseldorf%2C_December_2020_%2809%29.jpg)
# 
# Kürschner (talk) 17:51, 1 December 2020 (UTC), CC0, via Wikimedia Commons

# In[1]:


# install Pyomo and solvers
import requests
import imp

url = "https://raw.githubusercontent.com/jckantor/MO-book/main/python/helper.py"
helper = imp.new_module("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_glpk()


# ## The problem
# 
# There is an opportunity to operate a pop-up shop to sell a unique commemorative item for events held at a famous location. The items cost 12 &euro; each and will selL for 40 &euro;. Unsold items can be returned to the supplier at a value of only 2 &euro; due to their commemorative nature.
# 
# | Parameter | Symbol | Value |
# | :---: | :---: | :---: |
# | sales price | $r$ | 40 &euro; |
# | unit cost | $c$ | 12 &euro; |
# | salvage value | $w$ | 2 &euro; |
# 
# Profit will increase with sales. Demand for these items, however, will be high only if the weather is good. Historical data suggests the following scenarios.

# | Scenario ($s$) | Demand ($d_s$) | Probability ($p_s$) |
# | :---: | :-----: | :----------: |
# | Sunny Skies | 650 | 0.10 |
# | Good Weather | 400 | 0.60 |
# | Poor Weather | 200 | 0.30 |
# 
# The problem is to determine how many items to order for the pop-up shop. 
# 
# The dilemma is that the weather won't be known until after the order is placed. Ordering enough items to meet demand for a good weather day results in a  financial penalty on returned goods if the weather is poor. But ordering just enough items to satisfy demand on a poor weather day leaves "money on the table" if the weather is good.
# 
# How many items should be ordered for sale?

# ## Expected value for the mean scenario (EVM)
#  
# A naive solution to this problem is to place an order equal to the expected demand. The expected demand is given by
# 
# $$
# \begin{align*}
# \mathbb E[D] & = \sum_{s\in S} p_s d_s 
# \end{align*}
# $$
# 
# Choosing an order size $x = \mathbb E[d]$ results in an expected profit we call the **expected value of the mean scenario (EVM)**. 
# 
# Variable $y_s$ is the actual number of items sold if scenario $s$ should occur. The number sold is the lesser of the demand $d_s$ and the order size $x$.
# 
# $$
# \begin{align*}
# y_s & = \min(d_s, x) & \forall s \in S
# \end{align*}
# $$
# 
# Any unsold inventory $x - y_s$ remaining after the event will be sold at the salvage price $w$. Taking into account the revenue from sales $r y_s$, the salvage value of the unsold inventory $w(x - y_s)$, and the cost of the order $c x$, the profit $f_s$ for scenario $s$ is given by
# 
# $$
# \begin{align*}
# f_s & = r y_s + w (x - y_s) - c  x & \forall s \in S
# \end{align*}
# $$
# 
# The average or expected profit is given by
# 
# $$
# \begin{align*}
# \text{EVM} = \mathbb E[f] & = \sum_{s\in S} p_s f_s
# \end{align*}
# $$
# 
# These calculations can be executed using operations on the pandas dataframe. Let's begin by calculating the expected demand.
# 
# Below we create a pandas DataFrame object to store the scenario data.

# In[2]:


import numpy as np
import pandas as pd

# price information
r = 40
c = 12
w = 2

# scenario information
scenarios = {
    "sunny skies" : {"probability": 0.10, "demand": 650},
    "good weather": {"probability": 0.60, "demand": 400},
    "poor weather": {"probability": 0.30, "demand": 200},
}

df = pd.DataFrame.from_dict(scenarios).T
display(df)


# In[3]:


expected_demand = sum(df["probability"] * df["demand"])
print(f"Expected demand = {expected_demand}")


# Subsequent calculations can be done directly withthe  pandas dataframe holding the scenario data.

# In[4]:


df["order"] = expected_demand
df["sold"] = df[["demand", "order"]].min(axis=1)
df["salvage"] = df["order"] - df["sold"]
df["profit"] = r * df["sold"] + w * df["salvage"] - c * df["order"]

EVM = sum(df["probability"] * df["profit"])

print(f"Mean demand = {expected_demand}")
print(f"Expected value of the mean demand (EVM) = {EVM}")
display(df)


# ## Expected value of the stochastic solution (EVSS)
# 
# The optimization problem is to find the order size $x$ that maximizes expected profit subject to operational constraints on the decision variables. The variables $x$ and $y_s$ are non-negative integers, while $f_s$ is a real number that can take either positive and negative values. The number of goods sold in scenario $s$ has to be less than the order size $x$ and customer demand $d_s$. 
# 
# The problem to be solved is
# 
# $$
# \begin{align*}
# \text{EV} = & \max_{x, y_s} \mathbb E[F] = \sum_{s\in S} p_s f_s \\
# \text{subject to:} \\
# f_s & = r y_s + w(x - y_s) - c x & \forall s \in S\\
# y_s & \leq x & \forall s \in S \\
# y_s & \leq d_s & \forall s \in S
# \end{align*}
# $$
# 
# where $S$ is the set of all scenarios under consideration.

# In[5]:


import pyomo.environ as pyo
import pandas as pd

# price information
r = 40
c = 12
w = 2  

# scenario information
scenarios = {
    "sunny skies" : {"demand": 650, "probability": 0.1},
    "good weather": {"demand": 400, "probability": 0.6},
    "poor weather": {"demand": 200, "probability": 0.3},
}

# create model instance
m = pyo.ConcreteModel('Pop-up Shop')

# set of scenarios
m.S = pyo.Set(initialize=scenarios.keys())

# decision variables
m.x = pyo.Var(domain=pyo.NonNegativeIntegers)
m.y = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)
m.f = pyo.Var(m.S, domain=pyo.Reals)

# objective
@m.Objective(sense=pyo.maximize)
def EV(m):
    return sum([scenarios[s]["probability"]*m.f[s] for s in m.S])

# constraints
@m.Constraint(m.S)
def profit(m, s):
    return m.f[s] == r*m.y[s] + w*(m.x - m.y[s]) - c*m.x

@m.Constraint(m.S)
def sales_less_than_order(m, s):
    return m.y[s] <= m.x

@m.Constraint(m.S)
def sales_less_than_demand(m, s):
    return m.y[s] <= scenarios[s]["demand"]

# solve
solver = pyo.SolverFactory('glpk')
results = solver.solve(m)

# display solution using Pandas
print("Solver Termination Condition:", results.solver.termination_condition)
print("Expected Profit:", m.EV())
print()
for s in m.S:
    scenarios[s]["order"] = m.x()
    scenarios[s]["sold"] = m.y[s]()
    scenarios[s]["salvage"] = m.x() - m.y[s]()
    scenarios[s]["profit"] = m.f[s]()
    
df = pd.DataFrame.from_dict(scenarios).T
display(df)


# Optimizing over all scenarios provides an expected profit of 8,920 &euro;, an increase of 581 &euro; over the base case of simply ordering the expected number of items sold. The new solution places a larger order. In poor weather conditions there will be more returns and lower profit that is more than compensated by the increased profits in good weather conditions. 
# 
# The addtional value that results from solve of this planning problem is called the **Value of the Stochastic Solution (VSS)**. The value of the stochastic solution is the additional profit compared to ordering to meet expected in demand. In this case,
# 
# $$\text{VSS} = \text{EV} - \text{EVM} = 8,920 - 8,339 = 581$$

# ## Expected value with perfect information (EVPI)
# 
# Maximizing expected profit requires the size of the order be decided before knowing what scenario will unfold. The decision for $x$ has to be made "here and now" with probablistic information about the future, but without specific information on which future will actually transpire.
# 
# Nevertheless, we can perform the hypothetical calculation of what profit would be realized if we could know the future. We are still subject to the variability of weather, what is different is we know what the weather will be at the time the order is placed. 
# 
# The resulting value for the expected profit is called the **Expected Value of Perfect Information (EVPI)**.  The difference EVPI - EV is the extra profit due to having perfect knowledge of the future.
# 
# To compute the expected profit with perfect information, we let the order variable $x$ be indexed by the subsequent scenario that will unfold. Given decision varaible $x_s$, the model for EVPI becomes
# 
# $$
# \begin{align*}
# \text{EVPI} = & \max_{x_s, y_s} \mathbb E[f] = \sum_{s\in S} p_s f_s \\
# \text{subject to:} \\
# f_s & = r y_s + w(x_s - y_s) - c x_s & \forall s \in S\\
# y_s & \leq x_s & \forall s \in S \\
# y_s & \leq d_s & \forall s \in S
# \end{align*}
# $$
# 
# The following implementation is a variation of the prior cell.

# In[6]:


import pyomo.environ as pyo
import pandas as pd

# price information
r = 40
c = 12
w = 2  

# scenario information
scenarios = {
    "sunny skies" : {"demand": 650, "probability": 0.1},
    "good weather": {"demand": 400, "probability": 0.6},
    "poor weather": {"demand": 200, "probability": 0.3},
}

# create model instance
m = pyo.ConcreteModel('Pop-up Shop')

# set of scenarios
m.S = pyo.Set(initialize=scenarios.keys())

# decision variables
m.x = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)
m.y = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)
m.f = pyo.Var(m.S, domain=pyo.Reals)

# objective
@m.Objective(sense=pyo.maximize)
def EV(m):
    return sum([scenarios[s]["probability"]*m.f[s] for s in m.S])

# constraints
@m.Constraint(m.S)
def profit(m, s):
    return m.f[s] == r*m.y[s] + w*(m.x[s] - m.y[s]) - c*m.x[s]

@m.Constraint(m.S)
def sales_less_than_order(m, s):
    return m.y[s] <= m.x[s]

@m.Constraint(m.S)
def sales_less_than_demand(m, s):
    return m.y[s] <= scenarios[s]["demand"]

# solve
solver = pyo.SolverFactory('glpk')
results = solver.solve(m)

# display solution using Pandas
print("Solver Termination Condition:", results.solver.termination_condition)
print("Expected Profit:", m.EV())
print()
for s in m.S:
    scenarios[s]["order"] = m.x[s]()
    scenarios[s]["sold"] = m.y[s]()
    scenarios[s]["salvage"] = m.x[s]() - m.y[s]()
    scenarios[s]["profit"] = m.f[s]()
    
df = pd.DataFrame.from_dict(scenarios).T
display(df)


# ## Summary
# 
# To summarize, have computed three different solutions to the problem of order size:
# 
# * The expected value of the mean solution (EVM) is the expected profit resulting from ordering the number of items expected to sold under all scenarios. 
# 
# * The expected value of the stochastic solution (EVSS) is the expected profit found by solving an two-state optimization problem where the order size was the "here and now" decision without specific knowledge of which future scenario would transpire.
# 
# * The expected value of perfect information (EVPI) is the result of a hypotherical case where knowledge of the future scenario was somehow available when then order had to be placed.  
# 
# For this example we found
# 
# | Solution | Value (&euro;) |
# | :------  | ----: |
# | Expected Value of the Mean Solution (EVM) | 8,399.0 | 
# | Expected Value of the Stochastic Solution (EVSS) | 8,920.0 |
# | Expected Value of Perfect Information (EVPI) | 10,220.0 |
# 
# These results verify our expectation that
# 
# $$
# \begin{align*}
# EVM \leq EVSS \leq EVPI
# \end{align*}
# $$
# 
# The value of the stochastic solution 
# 
# $$
# \begin{align*}
# VSS = EVSS - EVM = 581
# \end{align*}
# $$
# 
# The value of perfect information
# 
# $$
# \begin{align*}
# VPI = EVPI - EVSS = 1,300
# \end{align*}
# $$
# 
# 
# As one might expect, there is a cost that results from lack of knowledge about an uncertain future.

# In[ ]:




