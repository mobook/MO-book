#!/usr/bin/env python
# coding: utf-8

# # Making the Best of the Worst

# In[1]:


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

# In[99]:


import pandas as pd

BIM_scenarios = pd.DataFrame([
    [12, 9], 
    [11, 10], 
    [8, 11]], 
    columns=["product 1", "product 2"])

BIM_scenarios.index.name = "scenarios"
print("\nProfit scenarios")
display(BIM_scenarios)

BIM_resources = pd.DataFrame([
    ["silicon", 1000, 1, 0],
    ["germanium", 1500, 0, 1],
    ["plastic", 1750, 1, 1],
    ["copper", 4000, 1, 2]],
    columns = ["resource", "available", "product 1", "product 2"])
BIM_resources = BIM_resources.set_index("resource")
    
print("\nAvailable resources and resource requirements")
display(BIM_resources)
    


# ## Pyomo Model
# 
# An implementation of the maximum worst-case profit model.

# In[159]:


import pyomo.environ as pyo

def maxmin(scenarios, resources):
    
    model    = pyo.ConcreteModel('BIM')
    
    products = resources.columns.tolist()
    products.remove('available')
    
    model.I = pyo.Set(initialize=resources.index)
    model.J = pyo.Set(initialize=products)
    model.S = pyo.Set(initialize=scenarios.index)
    
    model.a = pyo.Param(model.I, model.J, rule = lambda model, i, j: resources.loc[i, j])
    model.b = pyo.Param(model.I, rule = lambda model, i: resources.loc[i, "available"])
    model.c = pyo.Param(model.S, model.J, rule = lambda model, s, j: scenarios.loc[s, j])
    
    model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.z  = pyo.Var() 
    
    @model.Objective(sense=pyo.maximize)
    def profit(model):
        return model.z
    
    @model.Constraint(model.S)
    def scenario_profit(m, s):
        return m.z <= sum(model.c[s, j] * model.x[j] for j in model.J)
        
    @model.Constraint(model.I)
    def resource_limits(model, i):
        return sum(model.a[i, j] * model.x[j] for j in model.J) <= model.b[i]

    return model

BIM = maxmin(BIM_scenarios, BIM_resources)
pyo.SolverFactory('glpk').solve(BIM)

worst_case_plan = pd.Series({j: BIM.x[j]() for j in BIM.J})
worst_case_profit = BIM.profit()

print("\nworst case profit = ", worst_case_profit)
print(f"\nworst case production plan:\n")
display(worst_case_plan)


# ## Is maximizing the worst case a good idea?
# 
# Maximizing the worst case among all scenarios may be a pessimistic planning outlook. It may be worth investigating alternative planning outlooks The first step is to create a model to optimize a single scenario. Without repeating the mathematical description, the following Pyomo model is simply the `maxmin` model adapted to a single scenario.

# In[160]:


def max_profit(scenario, resources):
    
    model    = pyo.ConcreteModel('BIM')
    
    products = resources.columns.tolist()
    products.remove('available')
    
    model.I = pyo.Set(initialize=resources.index)
    model.J = pyo.Set(initialize=products)
    
    model.a = pyo.Param(model.I, model.J, rule = lambda model, i, j: resources.loc[i, j])
    model.b = pyo.Param(model.I, rule = lambda model, i: resources.loc[i, "available"])
    model.c = pyo.Param(model.J, rule = lambda model, j: scenario[j])
    
    model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.z  = pyo.Var() 
    
    @model.Objective(sense=pyo.maximize)
    def profit(m, s):
        return sum(model.c[j] * model.x[j] for j in model.J)
        
    @model.Constraint(model.I)
    def resource_limits(model, i):
        return sum(model.a[i, j] * model.x[j] for j in model.J) <= model.b[i]

    return model


# ## Optimizing for the mean scenario
# 
# The next cell computes the optimal plan for the mean scenario.

# In[166]:


# create mean scenario

mean_case = max_profit(BIM_scenarios.mean(), BIM_resources)
pyo.SolverFactory('glpk').solve(mean_case)

mean_case_profit = mean_case.profit()
mean_case_plan = pd.Series({j: m.x[j]() for j in m.J})

print("\nmean case profit", mean_case_profit)
print("\nmean case production plan\n")
print(mean_case_plan)


# The expected profit under the mean scenario if 17,833, which is 333 greater than maximizing the worst case. Also note the change in production plan. Would plan should be preferred ... one that produces no less than 17,500 under all scenarios, or one that produces 17,833 when averaged over all scenarios?

# In[178]:


mean_outcomes = BIM_scenarios.dot(mean_case_plan)
worst_case_outcomes = BIM_scenarios.dot(worst_case_plan)
mean_outcomes.plot(kind="bar")


# In[179]:


worst_case_outcomes.plot(kind="bar")


# In[ ]:




