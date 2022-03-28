#!/usr/bin/env python
# coding: utf-8

# # Minimum Sum of Absolute Errors Regression
# 
# In the contemporary context of machine learning, linear regression is a supervised learning technique that produces a linear model to predict values of a dependent variable from known values of one or more independent variables. Linear regression has a long history dating back to at least the 19th century, and is a mainstay of modern data analysis.
# 
# This notebook demonstrates linear regression by using to linear programming to minimize a sum of absolute errors between the model prediction and data from a training set. The sum of absolute values of  errors is the $L_1$ norm which is known to have favorable robustness characteristics in practical use. We follow closely this [paper](https://www.jstor.org/stable/1402501).

# In[150]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## A Wine Quality Dataset
# 
# 
# ### Downloading the data set
# 
# Physical, chemical, and sensory quality properties were collected for a large number of red and white wines produced in the Portugal then donated to the UCI machine learning repository (Cortez, Paulo, Cerdeira, A., Almeida, F., Matos, T. & Reis, J.. (2009). Wine Quality. UCI Machine Learning Repository.) The following cell reads the data for red wines directly from the UCI machine learning repository.
# 
# Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision support systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016

# In[221]:


import pandas as pd
import matplotlib.pyplot as plt

red_wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
red_wines


# ### A preliminary look at the data
# 
# The data consists of 1,599 measurements of eleven physical and chemical characteristics plus an integer measure of sensory quality recorded on a scale from 3 to 8. Histograms provides insight into the values and variability of the data set.

# In[222]:


fig, ax = plt.subplots(3, 4, figsize=(12, 8), sharey=True)

for a, column in zip(ax.flatten(), wine_quality.columns):
    red_wines[column].hist(ax=a, bins=30)
    a.axvline(wine_quality[column].mean(), color='r', label="mean")
    a.set_title(column)
    a.legend()
    
plt.tight_layout()


# Here the "output" is the reported sensory quality.

# In[223]:


red_wines.corr()["quality"].plot(kind="bar", grid=True)


# In[224]:


ax = wine_quality.boxplot(column="alcohol", by="quality")


# ## Simple Line Fitting
# 
# $$
# \begin{align*}
# y_i & = a x_i + b + \epsilon_i& \forall i\in I
# \end{align*}
# $$
# 
# $$
# \begin{align*}
# \min \sum_{i\in I} \left| y_i - a x_i - b \right|
# \end{align*}
# $$
# 

# In[225]:


import pyomo.environ as pyo

def l1_fit_version1(df, y_col, x_col):

    m = pyo.ConcreteModel("L1 Regression Model")

    m.I = pyo.RangeSet(len(red_wines))

    @m.Param(m.I)
    def y(m, i):
        return df.loc[i-1, y_col]

    @m.Param(m.I)
    def X(m, i):
        return df.loc[i-1, x_col]

    # regression
    m.a = pyo.Var()
    m.b = pyo.Var(domain=pyo.Reals)

    m.e_pos = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.e_neg = pyo.Var(m.I, domain=pyo.NonNegativeReals)
                            
    @m.Expression(m.I)
    def prediction(m, i):
        return m.a * m.X[i] + m.b

    @m.Constraint(m.I)
    def prediction_error(m, i):
        return m.e_pos[i] - m.e_neg[i] == m.prediction[i] - m .y[i]

    @m.Objective(sense=pyo.minimize)
    def sum_of_absolute_values(m):
        return sum(m.e_pos[i] + m.e_neg[i] for i in m.I)

    pyo.SolverFactory('cbc').solve(m)
    
    return m

m = l1_fit_version1(red_wines, "quality", "alcohol")
print(m.sum_of_absolute_values())


# In[227]:


vars = {i: l1_fit_version1(red_wines, "quality", i).sum_of_absolute_values() for i in red_wines.columns}


# In[229]:


pd.Series(vars).plot(kind="bar")


# In[219]:


red_wines["prediction"] = [m.prediction[i]() for i in m.I]
red_wines["quality"].hist(label="data")

red_wines.plot(x="quality", y="prediction", kind="scatter")


# ## Multivariable Regression

# In[212]:


import pyomo.environ as pyo

def l1_fit_version2(df, y_col, x_cols):

    m = pyo.ConcreteModel("L1 Regression Model")

    m.I = pyo.RangeSet(len(red_wines))
    m.J = pyo.Set(initialize=x_cols)

    @m.Param(m.I)
    def y(m, i):
        return df.loc[i-1, y_col]

    @m.Param(m.I, m.J)
    def X(m, i, j):
        return df.loc[i-1, j]

    # regression
    m.a = pyo.Var(m.J)
    m.b = pyo.Var(domain=pyo.Reals)

    m.e_pos = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.e_neg = pyo.Var(m.I, domain=pyo.NonNegativeReals)
                            
    @m.Expression(m.I)
    def prediction(m, i):
        return sum(m.a[j] * m.X[i, j] for j in m.J) + m.b

    @m.Constraint(m.I)
    def prediction_error(m, i):
        return m.e_pos[i] - m.e_neg[i] == m.prediction[i] - m.y[i]

    @m.Objective(sense=pyo.minimize)
    def sum_of_absolute_values(m):
        return sum(m.e_pos[i] + m.e_neg[i] for i in m.I)

    pyo.SolverFactory('cbc').solve(m)
    
    return m

m = l1_fit_version2(red_wines, "quality", 
                    ["alcohol", "volatile acidity", "citric acid", "sulphates", \
                     "total sulfur dioxide", "density", "fixed acidity"])
print(m.sum_of_absolute_values())

for j in m.J:
    print(f"{j}  {m.a[j]()}")

red_wines["prediction"] = [m.prediction[i]() for i in m.I]
red_wines["quality"].hist(label="data")

red_wines.plot(x="quality", y="prediction", kind="scatter")


# In[67]:


import pyomo.environ as pyo

l1 = pyo.ConcreteModel("L1 Regression Model")

l1.I = pyo.RangeSet(len(wine_quality))
l1.J = pyo.Set(initialize=["alcohol"])

@l1.Param(l1.I)
def y(l1, i):
    return wine_quality.loc[i-1, "quality"]

@l1.Param(l1.I, l1.J)
def X(l1, i, j):
    return wine_quality.loc[i-1, j]

l1.m = pyo.Var()
l1.b = pyo.Var()

l1.e_pos = pyo.Var(l1.I, domain=pyo.NonNegativeReals)
l1.e_neg = pyo.Var(l1.I, domain=pyo.NonNegativeReals)

@l1.Constraint(l1.I)
def regression(l1, i):
    return l1.e_pos[i] - l1.e_neg[i] == l1.y[i] - l1.X[i]*l1.m - l1.b

@l1.Objective()
def sum_of_absolute_values(l1):
    return sum(l1.e_pos[i] + l1.e_neg[i] for i in l1.I)

pyo.SolverFactory('cbc').solve(l1)

print(l1.m(), l1.b())
wine_quality["prediction"] = l1.m()*wine_quality["alcohol"] + l1.b()
wine_quality["prediction error"] = wine_quality["prediction"] - wine_quality["quality"]

#ax = wine_quality.plot(x="alcohol", y=["quality"], kind="scatter")
#wine_quality.plot(x="alcohol", y=["prediction"], ax=ax)

ax = wine_quality["quality"].hist()
wine_quality["prediction"].hist(ax=ax)


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




