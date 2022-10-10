#!/usr/bin/env python
# coding: utf-8

# # Fractional BIM

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/jckantor/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_glpk()


# In[2]:


import pyomo.environ as pyo


# In[3]:


def BIM_with_revenues_minus_costs():
    
    model    = pyo.ConcreteModel('BIM')
    
    model.x1 = pyo.Var(domain=pyo.NonNegativeReals)
    model.x2 = pyo.Var(domain=pyo.NonNegativeReals)

    model.revenue       = pyo.Expression( expr = 12*model.x1  +   9*model.x2 )
    model.variable_cost = pyo.Expression( expr = 7/6*model.x1 + 5/6*model.x2 )
    model.fixed_cost    = 100

    model.profit    = pyo.Objective( sense= pyo.maximize
      , expr = model.revenue - model.variable_cost - model.fixed_cost )

    model.silicon   = pyo.Constraint(expr =    model.x1              <= 1000)
    model.germanium = pyo.Constraint(expr =                 model.x2 <= 1500)
    model.plastic   = pyo.Constraint(expr =    model.x1 +   model.x2 <= 1750)
    model.copper    = pyo.Constraint(expr =  4*model.x1 + 2*model.x2 <= 4800)

    return model


# In[4]:


def BIM_with_revenues_over_costs():
    model    = pyo.ConcreteModel('BIM')
    
    model.y1 = pyo.Var(within=pyo.NonNegativeReals)
    model.y2 = pyo.Var(within=pyo.NonNegativeReals)
    model.t  = pyo.Var(within=pyo.NonNegativeReals)

    model.revenue       = pyo.Expression( expr = 12*model.y1  +   9*model.y2 )
    model.variable_cost = pyo.Expression( expr = 7/6*model.y1 + 5/6*model.y2 )
    model.fixed_cost    = 100

    model.profit    = pyo.Objective( sense= pyo.maximize
                                   , expr = model.revenue)

    model.silicon   = pyo.Constraint(expr =    model.y1              <= 1000*model.t)
    model.germanium = pyo.Constraint(expr =                 model.y2 <= 1500*model.t)
    model.plastic   = pyo.Constraint(expr =    model.y1 +   model.y2 <= 1750*model.t)
    model.copper    = pyo.Constraint(expr =  4*model.y1 + 2*model.y2 <= 4800*model.t)
    model.frac      = pyo.Constraint(expr = model.variable_cost+model.fixed_cost*model.t == 1 )
    
    return model


# In[5]:


BIM_linear = BIM_with_revenues_minus_costs()
results = pyo.SolverFactory('glpk').solve(BIM_linear)

print('X=({:.1f},{:.1f}) value={:.3f} revenue={:.3f} cost={:.3f}'.format(
    pyo.value(BIM_linear.x1),
    pyo.value(BIM_linear.x2),
    pyo.value(BIM_linear.profit),
    pyo.value(BIM_linear.revenue),
    pyo.value(BIM_linear.variable_cost)+pyo.value(BIM_linear.fixed_cost)))


# In[6]:


BIM_fractional = BIM_with_revenues_over_costs()
results = pyo.SolverFactory('glpk').solve(BIM_fractional)
t = pyo.value(BIM_fractional.t)
print('X=({:.1f},{:.1f}) value={:.3f} revenue={:.3f} cost={:.3f}'.format(
    pyo.value(BIM_fractional.y1/t),
    pyo.value(BIM_fractional.y2/t),
    pyo.value(BIM_fractional.profit/(BIM_fractional.variable_cost+BIM_fractional.fixed_cost*t)),
    pyo.value(BIM_fractional.revenue/t),
    pyo.value(BIM_fractional.variable_cost/t)+pyo.value(BIM_fractional.fixed_cost)))


# In[ ]:




