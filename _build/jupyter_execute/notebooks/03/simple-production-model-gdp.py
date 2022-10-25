#!/usr/bin/env python
# coding: utf-8

# # Production Model with Disjunctions
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
# This notebook shows how to express disjunctions in Pyomo models using  the Generalized Disjunctive Programming (GDP) extension for a simple production model.
# 

# ## Production Model
# 
# A small production facility produces two products, $X$ and $Y$. With current technology $\alpha$, the facility is subject to under the following conditions and constraints:
# 
# * Product $X$ requires 1 hour of labor A, 2 hours of labor B, and \\$100 of raw material. Product $X$ sells for \\$270 per unit. The daily demand is limited to 40 units.
# 
# * Product $Y$ requires 1 hour of labor A, 1 hour of labor B, and \\$90 of raw material. Product $Y$ sells for \\$210 per unit with unlimited demand. 
# 
# * There are 50 hours per day of labor A available at a cost of \\$50/hour.
# 
# * There are 100 hours per day of labor B available at a cost off \\$40/hour.
# 
# Using the given data we see that net profit for each unit of $X$ and $Y$ is \\$40 and \\$30, respectively. The optimal product strategy is the solution to a linear program
# 
# $$
# \begin{align*}
# \max_{x, y \geq 0} &\quad \text{profit}
# \\
# \text{subject to:}\qquad\qquad
# \\
# \text{profit} & = 40 x + 30 y \\
# x & \leq 40 & \text{Demand}\\
# x + y & \leq 80 & \text{Labor A} \\
# 2 x + y & \leq 100 & \text{Labor B} \\
# \end{align*}
# $$
# 
# Now suppose a new technology $\beta$ is available that affects that lowers the cost of product $X$. With the new technology, only 1.5 hours of labor B is required per unit of $X$.
# 
# 270 - 100 - 50 - 2*40 = $40
# 
# 270 - 100 - 50 - 1.5*40 = $60

# In a machine scheduling problem, for example, the choice may be to start one job ("A") either before or after a different job ("B"), where $\tau_A$ and $\tau_B$ denote the start time of the jobs. Since one or the other of the two constraints must hold, but not both, this situation corresponds to an exclusive-or disjunction of  the two constraints represented as
# 
# $$ \underbrace{\left[\tau_A \leq \tau_B\right]}_\alpha \veebar \underbrace{\left[\tau_A \geq \tau_B\right]}_\beta$$
# 
# There are several commonly used techniques for embedding disjunctions into mixed-integer linear programs. The "big-M" technique introduces a binary decision variable for every exclusive-or disjunction between two constraints. In this case, let
# 
# 
# 
# The notebook presents a very simple production model where there is a choice between a baasi

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# In[2]:


import pyomo.environ as pyo

model = pyo.ConcreteModel("Multi-Product Plant")

# decision variables
model.profit = pyo.Var()
model.production_x = pyo.Var(domain=pyo.NonNegativeReals)
model.production_y = pyo.Var(domain=pyo.NonNegativeReals)

# profit objective
@model.Objective(sense=pyo.maximize)
def maximize_profit(model):
    return  model.profit

# constraints
@model.Constraint()
def profit_expr(model):
    return model.profit == 40*model.production_x + 30*model.production_y

@model.Constraint()
def demand(model):
    return model.production_x <= 40

@model.Constraint()
def laborA(model):
    return model.production_x + model.production_y <= 80

@model.Constraint()
def laborB(model):
    return 2*model.production_x + model.production_y <= 100

# solve
pyo.SolverFactory('cbc').solve(model)

print(f"Profit = {model.profit()}")
print(f"Production X = {model.production_x()}")
print(f"Production Y = {model.production_y()}")


# ## Would a new technology improve profit?
# 
# Labor B is a relatively high cost for the production of product X.  A new technology has been developed with the potential to lower cost by reducing the time required to finish product X to 1.5 hours, but require a more highly skilled labor type C at a unit cost of $60 per hour. Would 
# 
# 
# $$
# \begin{align*}
# \max_{x, y \geq 0} &\quad \text{profit}\\
# \\
# \text{subject to:}\qquad\qquad
# \\
# x & \leq 40 & \text{Demand}\\
# x + y & \leq 80 & \text{Labor A} \\
# \\
# \begin{bmatrix}
# \text{profit} = 40x + 30y\\
# 2 x + y \leq 100
# \end{bmatrix}
# & \veebar
# \begin{bmatrix}
# \text{profit} = 60x + 30y\\
# 1.5 x + y \leq 100
# \end{bmatrix}
# \end{align*}
# $$

# In[3]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

model = pyo.ConcreteModel("Multi-Product Plant")

# decision variables
model.profit = pyo.Var(bounds=(-10000, 10000))
model.production_x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 200))
model.production_y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 200))

# profit objective
@model.Objective(sense=pyo.maximize)
def maximize_profit(model):
    return  model.profit

@model.Constraint()
def demand(model):
    return model.production_x <= 40

@model.Constraint()
def laborA(model):
    return model.production_x + model.production_y <= 80

@model.Disjunct()
def technology_A(disjunct):
    model = disjunct.model()
    disjunct.laborB = \
        pyo.Constraint(expr = 2*model.production_x + model.production_y <= 100)
    disjunct.profit_expr = \
        pyo.Constraint(expr = model.profit == 40*model.production_x + 30*model.production_y)

@model.Disjunct()
def technology_B(disjunct):
    model = disjunct.model()
    disjunct.laborB = \
        pyo.Constraint(expr = 1.5*model.production_x + model.production_y <= 100)
    disjunct.profit_expr = \
        pyo.Constraint(expr = model.profit == 60*model.production_x + 30*model.production_y)

@model.Disjunction(xor=True)
def technology(model):
    return [model.technology_A, model.technology_B]

# solve
pyo.TransformationFactory("gdp.bigm").apply_to(model)
pyo.SolverFactory('cbc').solve(model)

print(f"Profit = {model.profit()}")
print(f"Production X = {model.production_x()}")
print(f"Production Y = {model.production_y()}")


# In[4]:


model = pyo.ConcreteModel()
          
model.profit = pyo.Var(bounds=(-1000, 10000))
model.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1000))
model.y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1000))

@model.Objective(sense=pyo.maximize)
def maximize_profit(model):
    return model.profit

@model.Constraint()
def demand(model):
    return model.x <= 40

@model.Constraint()
def laborA(model):
    return model.x + model.y <= 80

@model.Disjunction(xor=True)
def technologies(model):
    return [[model.profit == 40*model.x + 30*model.y,
             2*model.x + model.y <= 100],
            
            [model.profit == 60*model.x + 30*model.y,
             1.5*model.x + model.y <= 100]]
            

pyo.TransformationFactory("gdp.bigm").apply_to(model)
pyo.SolverFactory('cbc').solve(model)

print(f"Profit = {model.profit()}")
print(f"x = {model.x()}")
print(f"y = {model.y()}")


# In[ ]:




