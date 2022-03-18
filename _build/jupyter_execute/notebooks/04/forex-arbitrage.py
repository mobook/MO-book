#!/usr/bin/env python
# coding: utf-8

# # FOREX Arbitrage
# 
# IN DEVELOPMENT

# In[1]:


# Import Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Cross-currency matrix
# 
# https://www.bloomberg.com/markets/currencies/cross-rates

# In[14]:


df = pd.DataFrame([[1.0, 0.5, 100], [2.0, 1.0, 1/0.0075], [0.01, 0.0075, 1.0]],
                  columns = ['USD', 'EUR', 'JPY'],
                  index = ['USD', 'EUR', 'JPY']).T

display(df)


# ## Simple Demonstration of Triangular Arbitrage
# 
# Consider the following cross-currency matrix. Entry $a_{m, n}$ is the number units of currency $m$ received in exchange for one unit of currency $n$.  We use the notation 
# 
# $$a_{m, n} = a_{m \leftarrow n}$$
# 
# as reminder of what the entries denote.

# In[3]:


# USD -> EUR -> USD
print(df.loc['USD', 'EUR'] * df.loc['EUR', 'USD'])

# USD -> JPY -> USD
print(df.loc['USD', 'JPY'] * df.loc['JPY', 'USD'])

# EUR -> JPY -> EUR
print(df.loc['EUR', 'JPY'] * df.loc['JPY', 'EUR'])


# There are no two way arbitrage opportunities. We check this by explicitly computing the exchange 
# 
# $$I \rightarrow J \rightarrow I$$
# 
# by computing
# 
# $$ a_{i \leftarrow j} \times a_{j \leftarrow i}$$
# 
# This particular example shows no net cost and no arbitrage for conversion from one currency to another and back again.
# 
# Now consider a three-way exchange
# 
# $$ I \rightarrow J \rightarrow K \rightarrow I $$
# 
# The exchange rate can be computed as
# 
# $$ a_{i \leftarrow k} \times a_{k \leftarrow j} \times a_{j \leftarrow i} $$
# 
# By direct caculation we see there is a three-way, or **triangular** arbitrage opportunity.

# In[4]:


I = 'USD'
J = 'JPY'
K = 'EUR'

print(df.loc[I, K] * df.loc[K, J] * df.loc[J, I])


# ## Modeling
# 
# The cross-currency table $A$ provides exchanges rates among currencies. The entry $a_{i,j}$ in row $i$, column $j$ tells us how many units of currency $i$ are received in exchange for one unit of currency $j$. We'll use the notation $a_{i,j} = a_{i\leftarrow j}$ to indicate this relationship.
# 
# We start with $w_j(0)$ units of currency $j \in N$. In the first phase of a transaction, an amount $x_{i\leftarrow j}(t)$  is committed to an exchange for currency $i$. After the commitment the unencumbered balance is
# 
# $$w_j(t-1) - \sum_{i\ne j} x_{i\leftarrow j}(t) \geq 0$$
# 
# which assumes no borrowing is allowed. 
# 
# The transaction is completed when the exchange credits accounts in the new currencies
# 
# $$ w_j(t) = w_j(t-1) - \underbrace{\sum_{i\ne j} x_{i\leftarrow j}(t)}_{\text{outgoing}} + \underbrace{\sum_{i\ne j} a_{j\leftarrow i}x_{j\leftarrow i}(t)}_{\text{incoming}} $$
# 
# The goal of this calculation is to find a set of transactions $x_{i\leftarrow j}(t) \geq 0$ such that $w_j(T) \geq w_j(0)$. 
# 
# 
# 
# 

# In[11]:


import pyomo.environ as pyo
import numpy as np


def arbitrage(T, df):

    m = pyo.ConcreteModel()

    # length of trading chain
    m.T0 = pyo.RangeSet(0, T)

    # number of transactions
    m.T1 = pyo.RangeSet(1, T)

    # currency *nodes*
    m.NODES = pyo.Set(initialize=df.index)

    # paths between currency nodes i -> j
    m.ARCS = pyo.Set(initialize = m.NODES * m.NODES, filter = lambda arb, i, j: i != j)

    # w[i, t] amount of currency n on hand at time t
    m.w = pyo.Var(m.NODES, m.T0, domain=pyo.NonNegativeReals)

    # x[m, n, t] amount of currency m converted to currency n in step t
    m.x = pyo.Var(m.ARCS, m.T1, domain=pyo.NonNegativeReals)

    # start with 100 units of a reserve currency in each currency account
    #@m.Constraint(m.NODES)
    #def initial_conditions(m, i):
    #    return m.w[i, 0] == 100.0/df.loc['USD', i]
    
    @m.Constraint()
    def initial_condition(m):
        return sum(m.w[j, 0] * df.loc['USD', j] for j in m.NODES) == 100.0

    @m.Constraint(m.NODES, m.T1)
    def max_trade(m, j, t):
        return m.w[j, t-1] >= sum(m.x[i, j, t] for i in m.NODES if i != j)

    @m.Constraint(m.NODES, m.T1)
    def balances(m, j, t):
        return m.w[j, t] == m.w[j, t-1] - sum(m.x[i, j, t] for i in m.NODES if i != j)                                + sum(df.loc[j, i]*m.x[j, i, t] for i in m.NODES if i != j)

    @m.Constraint(m.NODES)
    def arb_condition(m, j):
        return m.w[j, T] >= m.w[j, 0]

    @m.Objective(sense=pyo.maximize)
    def wealth(m):
        return sum(m.w[j, T] * df.loc['USD', j] for j in m.NODES)

    solver = pyo.SolverFactory('gurobi_direct')
    solver.solve(m)

    for t in m.T0:
        print(f"\nt = {t}\n")
        if t >= 1:
            for i, j in m.ARCS:
                if m.x[i,j,t]() > 0:
                    print(f"{j} -> {i}  Convert {m.x[i, j, t]()} {j} to {df.loc[i,j]*m.x[i,j,t]()} {i}")
            print()

        for n in m.NODES:
            print(f"w[{n},{t}] = {m.w[n, t]():9.2f} ")
            
arbitrage(3, df)


# ## Bloomberg FOREX data

# In[12]:


import pandas as pd
import io

bloomberg = """
	USD	EUR	JPY	GBP	CHF	CAD	AUD	HKD
USD	-	1.1096	0.0084	1.3148	1.0677	0.7915	0.7376	0.1279
EUR	0.9012	-	0.0076	1.1849	0.9622	0.7133	0.6647	0.1153
JPY	118.6100	131.6097	-	155.9484	126.6389	93.8816	87.4867	15.1724
GBP	0.7606	0.8439	0.0064	-	0.8121	0.6020	0.5610	0.0973
CHF	0.9366	1.0393	0.0079	1.2314	-	0.7413	0.6908	0.1198
CAD	1.2634	1.4019	0.0107	1.6611	1.3489	-	0.9319	0.1616
AUD	1.3557	1.5043	0.0114	1.7825	1.4475	1.0731	-	0.1734
HKD	7.8175	8.6743	0.0659	10.2784	8.3467	6.1877	5.7662	-
"""

df = pd.read_csv(io.StringIO(bloomberg.replace('-', '1.0')), sep='\t', index_col=0)
display(df)


# In[13]:


arbitrage(5, df)


# In[ ]:




