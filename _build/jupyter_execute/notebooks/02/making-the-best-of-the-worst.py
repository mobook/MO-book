#!/usr/bin/env python
# coding: utf-8

# # Making the Best of the Worst

# In[1]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Cost Data

# In[35]:


import pandas as pd

costs = pd.DataFrame([[12, 9], [11, 10], [8, 11]], columns=["c1", "c2"])
display(costs)


# ## Model

# In[34]:


import pyomo.environ as pyo

def BIM_maxmin(costs):
    model    = pyo.ConcreteModel('BIM')
    
    model.COSTS = pyo.Set(initialize=costs)
    
    model.x1 = pyo.Var(within=pyo.NonNegativeReals)
    model.x2 = pyo.Var(within=pyo.NonNegativeReals)
    model.z  = pyo.Var() 
    
    @model.Objective(sense=pyo.maximize)
    def profit(model):
        return model.z
    
    @model.Constraint(model.COSTS)
    def maxmin(model, c1, c2):
        return model.z <= c1 * model.x1 + c2 * model.x2
        
    @model.Constraint()
    def silicon(model):
        return model.x1 <= 1000
    
    @model.Constraint()
    def germanium(model):
        return model.x2 <= 1500
    
    @model.Constraint()
    def plastic(model):
        return model.x1 + model.x2 <= 1750
    
    @model.Constraint()
    def copper(model):
        return model.x1 + 2*model.x2 <= 4800

    return model


# In[36]:


BIM = BIM_maxmin(costs.to_numpy().tolist())
results = pyo.SolverFactory('glpk').solve(BIM)

print('X=({:.1f}, {:.1f}) value={:.3f}'.format(
    pyo.value(BIM.x1),
    pyo.value(BIM.x2),
    pyo.value(BIM.z)))


# In[ ]:




