#!/usr/bin/env python
# coding: utf-8

# # Production model using disjunctions

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Disjunctions
# 
# Disjunctions appear in applications where there is choice among discrete alternatives. Given two logical propositions $\alpha$ and $\beta$, the "or" disjunction is denoted by $\vee$ and defined by the truth table
# 
# | $\alpha$ | $\beta$ | $\alpha \vee \beta$ |
# | :-: | :-: | :-: |
# | False | False | False |
# | True | False | True |
# | False | True | True |
# | True | True | True |
# 
# The "exclusive or" is denoted by $\veebar$ and defined by the truth table
# 
# | $\alpha$ | $\beta$ | $\alpha \veebar \beta$ |
# | :-: | :-: | :-: |
# | False | False | False |
# | True | False | True |
# | False | True | True |
# | True | True | False |
# 
# This notebook shows how to express disjunctions in Pyomo models using the Generalized Disjunctive Programming (GDP) extension for a simple production model.
# 

# ## Multi-product factory optimization
# 
# A small production facility produces two products, $X$ and $Y$. With current technology $\alpha$, the facility is subject to the following conditions and constraints:
# 
# * Product $X$ requires 1 hour of labor A, 2 hours of labor B, and 100€ of raw material. Product $X$ sells for 270€ per unit. The daily demand is limited to 40 units.
# 
# * Product $Y$ requires 1 hour of labor A, 1 hour of labor B, and 90€ of raw material. Product $Y$ sells for 210€ per unit with unlimited demand. 
# 
# * There are 80 hours per day of labor A available at a cost of 50€/hour.
# 
# * There are 100 hours per day of labor B available at a cost of 40€/hour.
# 
# Using the given data we see that the net profit for each unit of $X$ and $Y$ is 40€ and 30€, respectively. The optimal product strategy is the solution to a linear program
# 
# $$
# \begin{align*}
# \max_{x, y \geq 0} \quad & \text{profit}\\
# \text{s.t.} \quad 
# & \text{profit}  = 40 x + 30 y\\
# & x  \leq 40 & \text{(demand)}\\
# & x + y  \leq 80 & \text{(labor A)} \\
# & 2 x + y  \leq 100 & \text{(labor B)}
# \end{align*}
# $$
# 

# In[2]:


import pyomo.environ as pyo

m = pyo.ConcreteModel("Multi-Product Factory")

# decision variables
m.profit = pyo.Var()
m.production_x = pyo.Var(domain=pyo.NonNegativeReals)
m.production_y = pyo.Var(domain=pyo.NonNegativeReals)

# profit objective
@m.Objective(sense=pyo.maximize)
def maximize_profit(model):
    return  m.profit

# constraints
@m.Constraint()
def profit_expr(model):
    return m.profit == 40*m.production_x + 30*m.production_y

@m.Constraint()
def demand(model):
    return m.production_x <= 40

@m.Constraint()
def laborA(model):
    return m.production_x + m.production_y <= 80

@m.Constraint()
def laborB(model):
    return 2*m.production_x + m.production_y <= 100

pyo.SolverFactory('cbc').solve(m)

print(f"Profit = {m.profit():.2f} €")
print(f"Production X = {m.production_x()}")
print(f"Production Y = {m.production_y()}")


# Now suppose a new technology $\beta$ is available that affects that lowers the cost of product $X$. With the new technology, only 1.5 hours of labor B is required per unit of $X$.
# 
# The net profit for unit of product $X$ with technology $\alpha$ is equal to $270 - 100 - 50 - 2 \cdot 40 = 40$€
# 
# The net profit for unit of product $X$ with technology $\beta$ is equal to $270 - 100 - 50 - 1.5 \cdot 40 = 60$€
# 
# The decision here is whether to use technology $\alpha$ or $\beta$. There are several commonly used techniques for embedding disjunctions into mixed-integer linear programs. The "big-M" technique introduces a binary decision variable for every exclusive-or disjunction between two constraints. 

# ## MILP implementation
# 
# Using MILP, we can formulate this problem as follows:
# 
# $$
# \begin{align*}
#     \max_{x, y \geq 0, z \in \mathbb{B}} \quad & \text{profit}\\
#     \text{s.t.} \quad 
#     & x  \leq 40 & \text{(demand)}\\
#     & x + y  \leq 80 & \text{(labor A)} \\
#     & \text{profit} \leq 40x + 30y + M z \\
#     & \text{profit} \leq 60x + 30y + M (1 - z) \\
#     & 2 x + y \leq 100  + M z \\ 
#     & 1.5 x + y \leq 100 + M (1 - z).
# \end{align*}
# $$
# 
# where the variable $z \in \{ 0, 1\}$ "activates" the constraints related to the old or new technology, respectively, and $M$ is a big enough number. The corresponding Pyomo implementation is given by:

# In[3]:


m = pyo.ConcreteModel("Multi-Product Factory - MILP")

# decision variables
m.profit = pyo.Var()
m.production_x = pyo.Var(domain=pyo.NonNegativeReals)
m.production_y = pyo.Var(domain=pyo.NonNegativeReals)
m.z = pyo.Var(domain=pyo.Binary)
M = 10000

# profit objective
@m.Objective(sense=pyo.maximize)
def maximize_profit(m):
    return  m.profit

# constraints
@m.Constraint()
def profit_constr_1(m):
    return m.profit <= 40*m.production_x + 30*m.production_y + M * m.z

@m.Constraint()
def profit_constr_2(m):
    return m.profit <= 60*m.production_x + 30*m.production_y + M * (1 - m.z)

@m.Constraint()
def demand(m):
    return m.production_x <= 40

@m.Constraint()
def laborA(m):
    return m.production_x + m.production_y <= 80

@m.Constraint()
def laborB_1(m):
    return 2*m.production_x + m.production_y <= 100 + M * m.z

@m.Constraint()
def laborB_2(m):
    return 1.5*m.production_x + m.production_y <= 100 + M * (1 - m.z)

pyo.SolverFactory('cbc').solve(m)

print(f"Profit = {m.profit():.2f} €")
print(f"Production X = {m.production_x()}")
print(f"Production Y = {m.production_y()}")


# ## Disjunctive programming implementation
# 
# Alternatively, we can formulate our problem using a disjunction, preserving the logical structure, as follows:
# 
# $$
# \begin{align*}
# \max_{x, y \geq 0} \quad & \text{profit}\\
# \text{s.t.} \quad 
# & x  \leq 40 & \text{(demand)}\\
# & x + y  \leq 80 & \text{(labor A)} \\
# & \begin{bmatrix}
#     \text{profit} = 40x + 30y\\
#     2 x + y \leq 100
# \end{bmatrix}
#  \veebar
# \begin{bmatrix}
#     \text{profit} = 60x + 30y\\
#     1.5 x + y \leq 100
#     \end{bmatrix}
# \end{align*}
# $$
# 
# This formulation, if allowed by the software at hand, has the benefit that the software can smartly divide the solution of this problem into sub-possibilities depending on the disjunction. Pyomo natively supports disjunctions, as illustrated in the following implementation:

# In[4]:


m = pyo.ConcreteModel()
          
m.profit = pyo.Var(bounds=(-1000, 10000))
m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1000))
m.y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1000))

@m.Objective(sense=pyo.maximize)
def maximize_profit(model):
    return m.profit

@m.Constraint()
def demand(model):
    return m.x <= 40

@m.Constraint()
def laborA(model):
    return m.x + m.y <= 80

@m.Disjunction(xor=True)
def technologies(model):
    return [[m.profit == 40*m.x + 30*m.y,
             2*m.x + m.y <= 100],
            
            [m.profit == 60*m.x + 30*m.y,
             1.5*m.x + m.y <= 100]]
            
pyo.TransformationFactory("gdp.bigm").apply_to(m)
pyo.SolverFactory('cbc').solve(m)

print(f"Profit = {m.profit():.2f} €")
print(f"x = {m.x()}")
print(f"y = {m.y()}")


# In[ ]:




