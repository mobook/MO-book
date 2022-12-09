#!/usr/bin/env python
# coding: utf-8

# # Kelly Criterion and Exponential Cones
# 
# The Kelly Criterion for the size of optimal bets was originally proposed in 1956 by John Kelly at Bell Laboratories. He identified an analogy between gambling and encoding information for transmission on noisy channels, then finding that by choosing bet sizes appropriately, then
# 
# > The maximum exponential rate of growth of he gambler's capital is equal to the rate of transmission of informration over the channel.
# 
# The idea actually pre-dates Kelly. In 1738, the famous mathematician Daniel Bernoulli offered a resolution to the St. Petersburg paradox (proposed by his cousin, Nicholas Bernoulli in 1711) by suggesting one should choose bets of investments with the highest geometric mean of outcomes.
# 
# As described by William Poundstone in the popular book "Fortune's Formula", following Kelly's paper, this idea this was quickly picked up by both gamblers and investors, with many colorful adventures by early adopters in both Las Vegas and Wall Street.
# 
# Following the analysis of Cajas (2021), this notebook presents a version of the the problem analysis demonstrating the applcation of exponential cone programming. 

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_mosek()
helper.install_gurobi()


# ## Problem
# 
# The classical presentation of the Kelly Criterion is to consider a betting proposition offering odds of $b$ for an outcome with propability $p$. In that case a bet pays off $b+1$ unit of money for every unit of money wagered. If we let $x$ represent the fraction of wealth that is wagered, then the gross returns are given by 
# 
# $$
# \begin{align}
# R_1 & = 1 + b w && p_1 = p \\
# R_2 & = 1 - w && p_2 = (1-p) \\
# \end{align}
# $$
# 
# The objective is to maximum the expected log return
# 
# $$
# \begin{align}
# \max_w p_1 \ln (R_1) + p_2 \ln (R_2) \\
# \end{align}
# $$
# 
# Follow Cajas (2021), we introduce decision variables $q_i$ where 
# $$
# \begin{align}
# q_i & \leq \ln(R_i) \\
# \exp(q_i) & \leq R_i \\
# \end{align}
# $$
# 
# The problem is now in the general form
# 
# $$
# \begin{align*}
# \max\quad & \sum_i p_i q_i \\
# \text{s.t.}\quad & \exp(q_i) \leq R_i & \forall i\in I\\
# & R_i = 1 + \sum_{j} b_{ij} w_j & \forall i\in I \\
# & \sum_j w_j \leq 1 \\
# \end{align*}
# $$
# 
# where the probabilities of each outcome, $p_i$ are known, and the gross returns $R_i$ are functions of the decision variables $w_j$.
# 
# A primal exponential cone is a convex set $K_{exp} = \{(x,y,r)\}$ such that i
# 
# $$y \exp(\frac{x}{y}) \leq r$$
# 
# and $y, r \geq 0$. With this notation, the Kelly problem becomes
# 
# $$
# \begin{align*}
# \max\quad & \sum_i p_i q_i \\
# \text{s.t.}\quad & (q_i, 1, R_i) \in K_{exp} & \forall i\in I\\
# & R_i = 1 + \sum_j b_{ij} w_j & \forall i\in I \\
# & \sum_j w_j \leq 1 \\
# \end{align*}
# $$
# 

# ## Example 1: Proposition Betting
# 
# $$
# \begin{align}
# \max_w&\quad p q_1 + (1-p) q_2 \\
# \text{s.t}\quad&\\
# & (q_1, 1, 1 + bw) \in K_{exp} \\
# & (q_2, 1, 1 - w) \in K_{exp} \\
# & w \geq 0
# \end{align}
# $$
# 
# **Programming Note:** The Pyomo Kernel Library includes exponential cone constraints, but reorders the arguments. As presented above, $K_{exp} = \{(x,y,r)\}$ refers to the convex cone described by
# 
# $$y \exp(\frac{x}{y}) \leq r$$
# 
# where $y, r \geq 0$. The Pyomo Kernel Library defines the same cone as 
# 
# $$x_1 \exp(\frac{x_2}{x_1}) \leq r$$
# 
# where $x_1, r \geq 0$. The concordance is then
# 
# $$
# \begin{align}
# x & \sim x_2 \\
# y & \sim x_1 \\
# r & \sim r
# \end{align}
# $$

# In[48]:


b = 1
p = 0.51


# In[49]:


import pyomo.kernel as pmo

m = pmo.block()

# decision variables
m.q1 = pmo.variable()
m.q2 = pmo.variable()
m.w = pmo.variable(lb=0)

# objective
m.lnR = pmo.objective(p*m.q1 + (1-p)*m.q2, sense=pmo.maximize)

# conic constraints
m.t1 = pmo.conic.primal_exponential.as_domain(1 + b*m.w, 1, m.q1)
m.t2 = pmo.conic.primal_exponential.as_domain(1 - m.w, 1, m.q2)

# solve with 'mosek_direct' or 'gurobi_direct'
pmo.SolverFactory('mosek_direct').solve(m)

m.lnR()


# ## Example 2: Luenberger's Betting Wheel
# 
# * https://www.nsm.buffalo.edu/~hassard/458/spreadsheets/betting_wheel.html
# * https://www.math.nyu.edu/~kohn/undergrad.finance/2003/hw2.pdf
# 

# In[ ]:


import pyomo.kernel as pmo

m = pmo.block()

# decision variables
m.q1 = pmo.variable()
m.q2 = pmo.variable()
m.q3 = pmo.variable()

m.w = pmo.variable(lb=0)
m.w = pmo.variable(lb=0)
m.w = pmo.variable(lb=0)

# objective
m.lnR = pmo.objective(p*m.q1 + (1-p)*m.q2, sense=pmo.maximize)

# conic constraints
m.t1 = pmo.conic.primal_exponential.as_domain(1 + b*m.w, 1, m.q1)
m.t2 = pmo.conic.primal_exponential.as_domain(1 - m.w, 1, m.q2)

# solve with 'mosek_direct' or 'gurobi_direct'
pmo.SolverFactory('mosek_direct').solve(m)

m.lnR()


# ## Example 3: Volatility Pumping with Index Funds
# 
# Following Luenberger Chapter 6, Thorp, Cajas.

# ## Example 4: Portfolio Design with Drawdown Constraints
# 
# Following Busseti, Boyd (2016)

# ## Relationship with Relative Entropy Optimization
# 
# Kullback-Leibler divergence
# 
# $$D_{KL}(P \| Q)$$

# ## Bibliographic Notes
# 
# The Kelly Criterion has been included in many tutorial introductions to finance and probability, and the subject of popular accounts.
# 
# > Poundstone, W. (2010). Fortune's formula: The untold story of the scientific betting system that beat the casinos and Wall Street. Hill and Wang. 
# 
# >  https://en.wikipedia.org/wiki/Kelly_criterion
# 
# The utility of conic programming to solve problems involving log growth is more recent. Here are some representative papers.
# 
# > Cajas, D. (2021). Kelly Portfolio Optimization: A Disciplined Convex Programming Framework. Available at SSRN 3833617. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3833617
# 
# > Busseti, E., Ryu, E. K., & Boyd, S. (2016). Risk-constrained Kelly gambling. The Journal of Investing, 25(3), 118-134. https://arxiv.org/pdf/1603.06183.pdf
# 
# 
# > Fu, A., Narasimhan, B., & Boyd, S. (2017). CVXR: An R package for disciplined convex optimization. arXiv preprint arXiv:1711.07582. https://arxiv.org/abs/1711.07582
# 
# 

# In[ ]:




