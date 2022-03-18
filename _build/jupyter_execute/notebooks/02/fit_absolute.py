#!/usr/bin/env python
# coding: utf-8

# # Example of minimizing a sum of absolute values
# 
# We use linear regression minimizing the absolute deviations. 
# 
# We follow closely this [paper](https://www.jstor.org/stable/1402501).
# 
# Let us start by adding `pyomo` and `glpk` to our `colab` session.

# In[ ]:


get_ipython().system('pip install -q pyomo')


# In[ ]:


# we add glpk now
get_ipython().system('sudo apt install libglpk-dev python3.8-dev libgmp3-dev')
get_ipython().system('apt-get install -y -qq glpk-utils')


# # Generate data

# In[ ]:


n_features = 1 # 1 may be visualized, > 1 works fine but no drawing
n_samples  = 1000
noise      = 30

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2020)
# generate regression dataset
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)

if n_features == 1:
    plt.scatter(X,y)
    plt.show()


# # Model
# 
# 

# In[ ]:


import pyomo.environ as pyo


# In[ ]:


m = pyo.ConcreteModel('min abs')


# In[ ]:


n,k = X.shape


# In[ ]:


m.I = range(n)
m.J = range(k)


# In[ ]:


m.ep = pyo.Var(m.I,within=pyo.NonNegativeReals)
m.em = pyo.Var(m.I,within=pyo.NonNegativeReals)
m.m  = pyo.Var(m.J)
m.b  = pyo.Var()


# In[ ]:


m.fit = pyo.Constraint( m.I, rule = lambda m, i : m.b + m.ep[i] - m.em[i] + sum(X[i][j]*m.m[j] for j in m.J) == y[i] )


# In[ ]:


m.obj = pyo.Objective( expr = sum( m.ep[i] + m.em[i] for i in m.I ), sense = pyo.minimize )


# In[ ]:


get_ipython().run_line_magic('time', "pyo.SolverFactory('glpk').solve(m)")


# In[ ]:


if n_features ==  1:
    plt.scatter(X,y)
    plt.plot([ x[0] for x in X ], [ pyo.value(m.b) + pyo.value(m.m[0])*x[0] for x in X ], 'r' )
    plt.show()
else:
    print( pyo.value(m.b),[pyo.value(m) for m in m.m.values() ] )

