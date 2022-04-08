#!/usr/bin/env python
# coding: utf-8

# # Making the Best of the Worst

# In[112]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Problem
# 
# A common formulation for to maximize profit of a multi-product facility in a resource constrained  environment is given by the linear program
# 
# $$
# \begin{align*}
# \max\ \text{profit}  = \sum_{j\in J} c_j x_j &\\
# \text{s.t.} \qquad \sum_{j\in J} a_{ij}x_j & \leq b_i & \forall i \in I\\
# x_j & \geq 0 & \forall j\in J
# \end{align*}
# $$
# 
# where $x_j$ is the production of product $j\in J$, $c_j$ is the net profit from producing and selling one unit of product $j$, $a_{i, j}$ is the amount of resource $i$ required to product a unit of product $j$, and $b_i$ is amount of resource $i\in I$ available. If this data is available, then the linear programming solution can provide a considerable of information regarding an optimal production plan and the marginal value of additional resources.
# 
# But what if coefficients of the model are uncertain? What should be the objective then? Does uncertainty change the production plan? Does the uncertainty change the marginal value assigned to resources? These are complex and thorny questions that will be largely reserved for later chapters of this book. However, it is possible to consider a specific situation within the current context.
# 
# Consider a situation where is of $S$ plausible models for the net profit. These might be a result of marketing studies or from considering plant operation under multiple scenarios. The set of profit models could be written
# 
# $$
# \begin{align*}
# \text{profit}_s & = \sum_{j} c_j^s x_j & \forall s\in S
# \end{align*}
# $$
# 
# where $s$ indexes the set of possible scenarios. The scenarios are all deemed equal, no probabilistic interpretation is given. 
# 
# One conservative criterion is to find maximize profit for the worst case. Letting $z$ denote the profit for the worst case, this criterion requires finding a solution for ${x_j}$ for ${j\in J}$ that satisfies
# 
# $$
# \begin{align*}
# \max_{x_j} z & \\
# \\
# \text{s.t.} \qquad \sum_{j\in J} c_j^s x_j & \geq z& \forall ss\in S\\
# \sum_{j\in J} a_{ij}x_j & \leq b_i & \forall i \in I\\
# x_j & \geq 0 & \forall j\in J
# \end{align*}
# $$
# 
# where $z$ is lowest profit that would be encountered under any condition.
# 

# ## Data

# In[115]:


import pandas as pd

BIM_profit_scenarios = pd.DataFrame([
    [12, 9], 
    [11, 10], 
    [8, 11]], 
    columns=["product 1", "product 2"])

BIM_profit_scenarios.index.name = "scenarios"

display(BIM_profit_scenarios)

BIM_resources = pd.DataFrame([
    ["silicon", 1000, 1, 0],
    ["germanium", 1500, 0, 1],
    ["plastic", 1750, 1, 1],
    ["copper", 4000, 1, 2]],
    columns = ["resource", "available", "product 1", "product 2"])
BIM_resources = BIM_resources.set_index("resource")
    
display(BIM_resources)
    


# ## Pyomo Model

# In[119]:


import pyomo.environ as pyo

def maximin(profit_scenarios, resources):
    
    model    = pyo.ConcreteModel('BIM')
    
    model.I = pyo.Set(initialize=resources.index)
    model.J = pyo.Set(initialize=profit_scenarios.columns)
    model.S = pyo.Set(initialize=profit_scenarios.index)
    
    model.a = pyo.Param(model.I, model.J, rule = lambda model, i, j: resources.loc[i, j])
    model.b = pyo.Param(model.I, rule = lambda model, i: resources.loc[i, "available"])
    model.c = pyo.Param(model.S, model.J, rule = lambda model, s, j: profit_scenarios.loc[s, j])
    
    model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.z  = pyo.Var() 
    
    @model.Objective(sense=pyo.maximize)
    def maxmin_profit(model):
        return model.z
    
    @model.Constraint(model.S)
    def scenario_profit(model, s):
        return model.z <= sum(model.c[s, j] * model.x[j] for j in model.J)
        
    @model.Constraint(model.I)
    def resource_limits(model, i):
        return sum(model.a[i, j] * model.x[j] for j in model.J) <= model.b[i]

    return model

BIM = BIM_maxmin(BIM_profit_scenarios, BIM_resources)
pyo.SolverFactory('glpk').solve(BIM)

results = pd.DataFrame([BIM.x[j]() for j in BIM.J], index=BIM.J, columns=["production"])
worst_case_profit = BIM.z()

print("worst case profit = ", worst_case_profit)
display(results)


# In[ ]:




