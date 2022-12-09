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

# ## Example 2: Luenberger's Betting Wheel
# 
# * https://www.nsm.buffalo.edu/~hassard/458/spreadsheets/betting_wheel.html
# * https://www.math.nyu.edu/~kohn/undergrad.finance/2003/hw2.pdf
# 

# ## Example 3: Volatility Pumping with Index Funds

# ## Example 4: Portfolio Design with Drawdown Constraints

# ## Relationship with Relative Entropy Optimization
# 
# Kullback-Leibler divergence
# 
# $$D_{KL}(P \| Q)$$

# ## Bibliographic Notes
# 
# > Cajas, D. (2021). Kelly Portfolio Optimization: A Disciplined Convex Programming Framework. Available at SSRN 3833617. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3833617
# 
# > Busseti, E., Ryu, E. K., & Boyd, S. (2016). Risk-constrained Kelly gambling. The Journal of Investing, 25(3), 118-134. https://arxiv.org/pdf/1603.06183.pdf
# 
# 
# > Fu, A., Narasimhan, B., & Boyd, S. (2017). CVXR: An R package for disciplined convex optimization. arXiv preprint arXiv:1711.07582. https://arxiv.org/abs/1711.07582
# 
# > Poundstone, W. (2010). Fortune's formula: The untold story of the scientific betting system that beat the casinos and Wall Street. Hill and Wang. 
