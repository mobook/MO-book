#!/usr/bin/env python
# coding: utf-8

# # Minimum Sum of Absolute Errors Regression
# 
# In the contemporary context of machine learning, linear regression is a supervised learning technique that produces a linear model to predict values of a dependent variable from known values of one or more independent variables. Linear regression has a long history dating back to at least the 19th century, and is a mainstay of modern data analysis.
# 
# This notebook demonstrates linear regression by using to linear programming to minimize a sum of absolute errors between the model prediction and data from a training set. The sum of absolute values of  errors is the $L_1$ norm which is known to have favorable robustness characteristics in practical use. We follow closely this [paper](https://www.jstor.org/stable/1402501).

# In[8]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# # Generate data

# In[34]:


from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

n_features = 1
n_samples = 1000
noise = 30

# generate regression dataset
np.random.seed(2020)
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)

if n_features == 1:
    plt.scatter(X, y)
    plt.show()


# # Model
# 
# 

# In[35]:


import pyomo.environ as pyo


# In[36]:


m = pyo.ConcreteModel('min abs')


# In[37]:


n, k = X.shape


# In[38]:


m.I = pyo.RangeSet(0, n-1)
m.J = pyo.RangeSet(0, k-1)


# In[39]:


m.ep = pyo.Var(m.I, domain=pyo.NonNegativeReals)
m.em = pyo.Var(m.I, domain=pyo.NonNegativeReals)
m.m = pyo.Var(m.J)
m.b = pyo.Var()


# In[40]:


@m.Constraint(m.I)
def fit(m, i):
    return y[i] == m.b + m.ep[i] - m.em[i] + sum(X[i][j]*m.m[j] for j in m.J)


# In[41]:


@m.Objective(sense=pyo.minimize)
def obj(m):
    return sum(m.ep[i] + m.em[i] for i in m.I)


# In[42]:


get_ipython().run_line_magic('time', "pyo.SolverFactory('glpk').solve(m)")


# In[43]:


if n_features ==  1:
    plt.scatter(X, y, alpha=0.5)
    plt.plot([x[0] for x in X], [pyo.value(m.b) + pyo.value(m.m[0])*x[0] for x in X ], 'r')
    plt.show()
else:
    print( pyo.value(m.b),[pyo.value(m) for m in m.m.values() ] )


# In[ ]:




