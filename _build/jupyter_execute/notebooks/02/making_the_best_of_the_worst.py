#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q pyomo')


# In[ ]:


# we add glpk now
get_ipython().system('sudo apt install libglpk-dev python3.8-dev libgmp3-dev')
get_ipython().system('apt-get install -y -qq glpk-utils')


# In[ ]:


import pyomo.environ as pyo


# In[ ]:


def BIM_maxmin( costs ):
    model    = pyo.ConcreteModel('BIM')
    
    model.x1 = pyo.Var(within=pyo.NonNegativeReals)
    model.x2 = pyo.Var(within=pyo.NonNegativeReals)
    model.z  = pyo.Var() 

    model.profit    = pyo.Objective( sense= pyo.maximize, expr = model.z )

    model.maxmin = pyo.ConstraintList()
    for (c1,c2) in costs:
        model.maxmin.add( expr = model.z <= c1*model.x1 + c2*model.x2 ) 

    model.silicon   = pyo.Constraint(expr =    model.x1              <= 1000)
    model.germanium = pyo.Constraint(expr =                 model.x2 <= 1500)
    model.plastic   = pyo.Constraint(expr =    model.x1 +   model.x2 <= 1750)
    model.copper    = pyo.Constraint(expr =  4*model.x1 + 2*model.x2 <= 4800)

    return model


# In[ ]:


BIM = BIM_maxmin( [[12,9], [11,10], [8, 11]] )
results = pyo.SolverFactory('glpk').solve(BIM)

print('X=({:.1f},{:.1f}) value={:.3f}'.format(
    pyo.value(BIM.x1),
    pyo.value(BIM.x2),
    pyo.value(BIM.z)))


# In[ ]:




