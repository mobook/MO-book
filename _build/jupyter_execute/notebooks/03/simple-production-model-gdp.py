#!/usr/bin/env python
# coding: utf-8

# # Production Model with Disjuncts

# In[2]:


# Import Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Production Model
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

# In[3]:


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

# In[4]:


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
    disjunct.laborB =         pyo.Constraint(expr = 2*model.production_x + model.production_y <= 100)
    disjunct.profit_expr =         pyo.Constraint(expr = model.profit == 40*model.production_x + 30*model.production_y)

@model.Disjunct()
def technology_B(disjunct):
    model = disjunct.model()
    disjunct.laborB =         pyo.Constraint(expr = 1.5*model.production_x + model.production_y <= 100)
    disjunct.profit_expr =         pyo.Constraint(expr = model.profit == 60*model.production_x + 30*model.production_y)

@model.Disjunction(xor=True)
def technology(model):
    return [model.technology_A, model.technology_B]

# solve
pyo.TransformationFactory("gdp.bigm").apply_to(model)
pyo.SolverFactory('cbc').solve(model)

print(f"Profit = {model.profit()}")
print(f"Production X = {model.production_x()}")
print(f"Production Y = {model.production_y()}")


# In[5]:


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




