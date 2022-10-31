#!/usr/bin/env python
# coding: utf-8

# # Wine quality prediction with L1 regression

# In[6]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Problem description
# 
# Regression is the task of fitting a model to data. If things go well, the model might provide useful predictions in response to new data. This notebook shows how linear programming and least absolute deviation (LAD) regression can be used to create a linear model for predicting wine quality based on physical and chemical properties. The example uses a well known data set from the machine learning community.
# 
# Physical, chemical, and sensory quality properties were collected for a large number of red and white wines produced in the Portugal then donated to the UCI machine learning repository (Cortez, Paulo, Cerdeira, A., Almeida, F., Matos, T. & Reis, J.. (2009). Wine Quality. UCI Machine Learning Repository.) The following cell reads the data for red wines directly from the UCI machine learning repository.
# 
# Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision support systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016
# 
# Let us first download the data

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
display(wines)


# ## Model objective: Mean Absolute Deviation (MAD)
# 
# Given $n$ repeated observations of a response variable $y_i$ (in this wine quality), the **mean absolute deviation** (MAD) of $y_i$ from the mean value $\bar{y}$ is
# 
# $$\text{MAD}\,(y) = \frac{1}{n} \sum_{i=1}^n | y_i - \bar{y}|$$
# 

# In[8]:


def MAD(df):
    return (df - df.mean()).abs().mean()

print("MAD = ", MAD(wines["quality"]))
      
fig, ax = plt.subplots(figsize=(12, 4))
ax = wines["quality"].plot(alpha=0.6, title="wine quality")
ax.axhline(wines["quality"].mean(), color='r', ls='--', lw=3)
  
mad = MAD(wines["quality"])
ax.axhline(wines["quality"].mean() + mad, color='g', lw=3)
ax.axhline(wines["quality"].mean() - mad, color='g', lw=3)
ax.legend(["y", "mean", "mad"])
ax.set_xlabel("observation")

plt.show()


# ## A preliminary look at the data
# 
# The data consists of 1,599 measurements of eleven physical and chemical characteristics plus an integer measure of sensory quality recorded on a scale from 3 to 8. Histograms provides insight into the values and variability of the data set.

# In[9]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharey=True)

for ax, column in zip(axes.flatten(), wines.columns):
    wines[column].hist(ax=ax, bins=30)
    ax.axvline(wines[column].mean(), color='r', label="mean")
    ax.set_title(column)
    ax.legend()
    
plt.tight_layout()


# ## Which features influence reported wine quality?
# 
# The art of regression is to identify the features that have explanatory value for a response of interest. This is where a person with deep knowledge of an application area, in this case an experienced onenologist will have a head start compared to the naive data scientist. In the absence of the experience, we proceed by examining the correlation among the variables in the data set.

# In[10]:


wines.corr()["quality"].plot(kind="bar", grid=True)


# In[11]:


wines[["volatile acidity", "density", "alcohol", "quality"]].corr()


# Collectively, these figures suggest `alcohol` is a strong correlate of `quality`, and several additional factors as  candidates for explanatory variables..

# ## LAD line fitting to identify features
# 
# An alternative approach is perform a series of single feature LAD regressions to determine which features have the largest impact on reducing the mean absolute deviations in the residuals.
# 
# $$
# \begin{align*}
# \min \frac{1}{I} \sum_{i\in I} \left| y_i - a x_i - b \right|
# \end{align*}
# $$
# 
# This computation has been presented in a prior notebook.

# In[12]:


import pyomo.environ as pyo

def lad_fit_1(df, y_col, x_col):

    m = pyo.ConcreteModel("L1 Regression Model")

    m.I = pyo.RangeSet(len(df))

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
    def mean_absolute_deviation(m):
        return sum(m.e_pos[i] + m.e_neg[i] for i in m.I)/len(m.I)

    pyo.SolverFactory('cbc').solve(m)
    
    return m

m = lad_fit_1(wines, "quality", "alcohol")

print(m.mean_absolute_deviation())


# This calculation is performed for all variables to determine which variables are the best candidates to explain deviations in wine quality.

# In[13]:


mad = (wines["alcohol"] - wines["alcohol"].mean()).abs().mean()
vars = {i: lad_fit_1(wines, "quality", i).mean_absolute_deviation() for i in wines.columns}

fig, ax = plt.subplots()
pd.Series(vars).plot(kind="bar", ax=ax, grid=True)
ax.axhline(mad, color='r', lw=3)
ax.set_title('mean absolute deviation following regression')


# In[14]:


wines["prediction"] = [m.prediction[i]() for i in m.I]
wines["quality"].hist(label="data")

wines.plot(x="quality", y="prediction", kind="scatter")


# ## Multivariate $L_1$-regression

# Let us now perform a full multivariate $L_1$-regression on the wine dataset to predict the wine quality $y$ using the provided wine features. We aim to find the coefficients $m_j$'s and $b$ that minimize the mean absolute deviation (MAD) by solving the following problem:
# 
# $$
# \begin{align*}
# \text{MAD}\,(\hat{y}) = \min_{m, \, b} \quad & \frac{1}{n} \sum_{i=1}^n | y_i - \hat{y}_i| \\
# \\
# \text{s. t.}\quad & \hat{y}_i = \sum_{j=1}^J x_{i, j} m_j + b & \forall i = 1, \dots, n,
# \end{align*}
# $$
# 
# where $x_{i, j}$ are values of 'explanatory' variables, i.e., the 11 physical and chemical characteristics of the wines. By taking care of the absolute value appearing in the objective function, this can be implemented in Pyomo as an LP as follows:

# In[20]:


import pyomo.environ as pyo

def l1_fit(df, y_col, x_cols):

    m = pyo.ConcreteModel("L1 Regression Model")

    m.I = pyo.RangeSet(len(df))
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
    def mean_absolute_deviation(m):
        return sum(m.e_pos[i] + m.e_neg[i] for i in m.I)/len(m.I)

    pyo.SolverFactory('cbc').solve(m)
    
    return m

m = l1_fit(wines, "quality", 
                    ["alcohol", "volatile acidity", "citric acid", "sulphates", \
                     "total sulfur dioxide", "density", "fixed acidity"])
print("MAD =",m.mean_absolute_deviation(),"\n")

for j in m.J:
    print(f"{j}  {m.a[j]()}")
print("\n")

wines["prediction"] = [m.prediction[i]() for i in m.I]
wines["quality"].hist(label="data")

wines.plot(x="quality", y="prediction", kind="scatter")
plt.show()


# ## How do these models perform?
# 
# A successful regression model would demonstrate a substantial reduction from $\text{MAD}\,(y)$ to $\text{MAD}\,(\hat{y})$. The value of $\text{MAD}\,(y)$ sets a benchmark for the regression. The linear regression model clearly has some capability to explain the observed deviations in wine quality. Tabulating the results of the regression using the MAD statistic we find
# 
# | Regressors | MAD |
# | :--- | ---: |
# | none | 0.683 |
# | alcohol only | 0.541 | 
# | all | 0.500 |
# 
# Are these models good enough to replace human judgment of wine quality? The reader can be the judge.
