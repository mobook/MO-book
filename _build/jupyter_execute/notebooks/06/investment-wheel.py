#!/usr/bin/env python
# coding: utf-8

# # Kelly Criterion and Optimal Growth Portfolios
# 
# The Kelly Criterion determines the size of a bet in a repeated game with binary outcomes. The analysis was proposed in 1956 by John Kelly at Bell Laboratories. Kelly identified an analogy between gambling on binary outcomes and Claude Shannon's work on encoding information for transmission on noisy channels. Kelly used the analogy to show
# 
# > The maximum exponential rate of growth of he gambler's capital is equal to the rate of transmission of information over the channel.
# 
# This idea actually predates Kelly. In 1738, for instance, Daniel Bernoulli offered a resolution to the St. Petersburg paradox previously proposed by his cousin, Nicholas Bernoulli. The resolution was to allocate bets among investments to produce the the highest geometric mean of returns. As popularized by William Poundstone in his book "Fortune's Formula", Kelly's analysis was quickly picked up by gamblers with colorful adventures in Las Vega by early adopters, but the result laid in obscurity among investors until much later.
# 
# Among the reasons why Kelly was neglected by investors were high profile critiques by the most famous economist of the 20th Century, Paul Samuelson. Samuelson objected on several grounds, among them is a lack of risk aversion that results in large bets and risky short term behavior, and that Kelly's result is applicable to only one of many utility functions that describe investor preferences. The controversy didn't end there, however, as other academic economists, including Harry Markowitz, and practitioners found ways to adapt the Kelly criterion to investment funds.
# 
# This notebook presents solutions to Kelly's problem and related applications the using exponential cones. A significant feature of this notebook is the the inclusion of a risk constraints recently proposed by Boyd and coworkers. These notes are based on recent papers such as Cajas (2021), Busseti, Ryu and Boyd (2016), Fu, Narasimhan, and Boyd (2017) and applied to examples presented by Luenberger (1999, 2013), and others. Additional bibliographic notes are provided at the end of the notebook.

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_mosek()


# ## Luenberger's Investment Wheel
# 
# ### Problem Statement
# 
# In Chapter 18 of his book "Investment Science", David Luenberger presents an "investment" wheel with multiple outcomes. The wheel is divided into sectors, each marked with a number. An investor can place a wager on each sector before the spin of the wheel.  After the wheel comes to rest, the investor receives a payout equal to the wager times the number that appears next to the marker. The game can be repeated any number of times.
# 
# ![](investment-wheel.png)
# 
# Given an initial wealth $W_0$, what is your investing strategy for repeated plays of the game?
# 
# * Is there an investing strategy that almost surely grows? 
# * What is the mean return for each spin of the wheel?
# 
# ### Modeling
# 
# The investment wheel is an example of a game with $n=3$ outcomes. For each outcome $n$ there is a payout $b_n$ that occurs with probability $p_n$ as shown in the following table.
# 
# | Outcome | Probability $p_n$ | Odds $b_n$ |
# | :--: | :--: | :--: |
# | A | 1/2 | 3 |
# | B | 1/3 | 2 |
# | C | 1/6 | 6 |
# 
# If a fraction of wealth $w_n$ is wagered on each outcome $n$, then the gross returns are given in the following table.
# 
# | Outcome | Probability $p_n$ | Gross Returns $R_n$ |
# | :--: | :--: | :--: |
# | A | 1/2 | 1 + 2$w_A$ - $w_B$ - $w_C$ |
# | B | 1/3 | 1 - $w_A$ + $w_B$ - $w_C$ |
# | C | 1/6 | 1 - $w_A$ - $w_B$ + 5$w_C$ |
# 
# For a general game with $n$ outcomes,the optimization problem is to maximize expected log return 
# 
# $$
# \begin{align}
# \max_{w \geq 0}\quad & \sum_{n\in N} p_n \log(R_n) \\
# \text{s.t.}\quad \\
# & \sum_{n\in N} w_n \leq 1 \\
# & R_n = 1 + b_n w_n - \sum_{n\in N} w_n & \forall n\in N \\
# \end{align}
# $$
# 
# 
# Reformulation of the objective function with exponential cones gives a conic program.
# 
# $$
# \begin{align}
# \max_{w\geq 0}\quad & \sum_{n\in N} p_n q_n \\
# \text{s.t.}\quad \\
# & \sum_{n\in N} w_n \leq 1 \\
# & R_n = 1 + b_n w_n - \sum_{n\in N} w_n & \forall n\in N \\
# & (R_n, 1, q_n) \in K_{exp} & \forall n \in N \\
# \end{align}
# $$
# 
# The solution of this program is shown below.

# In[5]:


# describe the betting wheel
sectors = {
    "A": {"p": 1/2, "b": 3},
    "B": {"p": 1/3, "b": 2},
    "C": {"p": 1/6, "b": 6},
}


# In[6]:


import pyomo.kernel as pmo

def wheel(sectors):

    m = pmo.block()
    
    S = sectors.keys()

    # decision variables
    m.w = pmo.variable_dict({s: pmo.variable(lb=0) for s in S})
    m.q = pmo.variable_dict({s: pmo.variable() for s in S})
    
    # objective
    m.ElogR = pmo.objective(sum(sectors[s]["p"]*m.q[s] for s in S), sense=pmo.maximize)
    
    # expression for returns
    m.R = pmo.expression_dict({s: pmo.expression(1 + sectors[s]["b"]*m.w[s] - sum(m.w[s] for s in S)) for s in S})
    
    # constraints
    m.sum_w = pmo.constraint(sum(m.w[s] for s in S) <= 1)
    m.conic = pmo.block_dict({s: pmo.conic.primal_exponential.as_domain(m.R[s], 1, m.q[s]) for s in S})

    # solve with 'mosek_direct' (conic) or 'ipopt' (nonlinear)
    pmo.SolverFactory('mosek_direct').solve(m)

    return m
        
m = wheel(sectors)

print(f"Expected Gross Return = {np.exp(m.ElogR()): 0.5f}\n")
for s in sectors.keys():
    print(f"Sector {s}:  p = {sectors[s]['p']:0.4f}   b = {sectors[s]['b']:0.2f}   w = {m.w[s]():0.5f}")


# ### Risk Constraints
# 
# For the risk constrained case, we consider a constraint
# 
# $$\mathbb{E}[R^{-\lambda}] \leq 1$$
# 
# where $\lambda$ is a risk aversion parameter.
# 
# $$\sum_{n\in N} p_n R_n^{-\lambda} \leq 1$$
# 
# Again
# 
# $$\sum_{n\in N} e^{\log(p_1) - \lambda\log(R_1)} \leq 1$$
# 
# Introducing $u_n \geq e^{\log(p_n) - \lambda q_n}$ using the $q_n$ defined above, we get
# 
# $$
# \begin{align}
# \sum_{n\in N} u_n & \leq 1 \\
# (u_n, 1, \log(p_n) - \lambda q_n) & \in K_{exp} & \forall n\in N \\
# \end{align}
# $$
# 
# The risk-constrained investment wheel is now
# 
# 
# $$
# \begin{align}
# \max_{w\geq 0}\quad & \sum_{n\in N} p_n q_n \\
# \text{s.t.}\quad \\
# & \sum_{n\in N} w_n \leq 1 \\
# &\sum_{n\in N} u_n  \leq 1 \\
# & R_n = 1 + b_n w_n - \sum_{n\in N} w_n & \forall n\in N \\
# & (R_n, 1, q_n) \in K_{exp} & \forall n \in N \\
# & (u_n, 1, \log(p_n) - \lambda q_n) \in K_{exp} & \forall n\in N \\
# \end{align}
# $$
# 

# In[7]:


import pyomo.kernel as pmo

def wheel_rsk(sectors, lambd=0):

    m = pmo.block()
    
    S = sectors.keys()

    # decision variables
    m.w = pmo.variable_dict({s: pmo.variable(lb=0) for s in S})
    m.q = pmo.variable_dict({s: pmo.variable() for s in S})
    
    # objective
    m.ElogR = pmo.objective(sum(sectors[s]["p"]*m.q[s] for s in S), sense=pmo.maximize)
    
    # expression for returns
    m.R = pmo.expression_dict({s: pmo.expression(1 + sectors[s]["b"]*m.w[s] - sum(m.w[s] for s in S)) for s in S})
    
    # constraints
    m.sum_w = pmo.constraint(sum(m.w[s] for s in S) <= 1)
    m.conic = pmo.block_dict({s: pmo.conic.primal_exponential.as_domain(m.R[s], 1, m.q[s]) for s in S})
    
    # risk constraints
    m.u = pmo.variable_dict({s: pmo.variable() for s in S})
    m.sum_u = pmo.constraint(sum(m.u[s] for s in S) <= 1)
    m.risk =  pmo.block_dict(
        {s: pmo.conic.primal_exponential.as_domain(m.u[s], 1, np.log(sectors[s]["p"]) - lambd*m.q[s]) for s in S}
    )

    # solve with 'mosek_direct' (conic) or 'ipopt' (nonlinear)
    pmo.SolverFactory('mosek_direct').solve(m)

    return m
        
m = wheel_rsk(sectors, 5)

print(f"Expected Gross Return = {np.exp(m.ElogR()): 0.5f}\n")
for s in sectors.keys():
    print(f"Sector {s}:  p = {sectors[s]['p']:0.4f}   b = {sectors[s]['b']:0.2f}   w = {m.w[s]():0.5f}")
    


# ### Effect of Risk Aversion
# 
# The following cell demonstrates the effect of increasing the risk aversion parameter $\lambda$.

# In[8]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(10, 4))

v = np.linspace(-1, 5)
lambd = np.exp(v)
results = [wheel_rsk(sectors, _) for _ in lambd]

ax[0].semilogx(lambd, [np.exp(m.ElogR()) for m in results])
ax[0].set_title('investment wheel return / spin')
ax[0].grid(True)

ax[1].semilogx(lambd, [[m.w[s]() for s in sectors.keys()] for m in results])
ax[1].set_title('investment weights')
ax[1].set_xlabel('risk aversion $\lambda$')
ax[1].legend([f"(p, b) = ({sectors[s]['p']:0.3f}, {sectors[s]['b']:0.1f})" for s in sectors.keys()],
            bbox_to_anchor=(1.05, 1.05))
ax[1].grid(True)
ax[1].set_ylim(0, 0.6)

fig.tight_layout()


# ### Exercises
# 
# 1. Is there a deterministic investment strategy for the investment wheel?. That is, is there investment strategy that provides a fixed return regardless of the outcome of the spin? Set up and solve a model to find that strategy.
# 
# 2. Find the variance in the outcome of the wheel, and plot the variance as a function of the risk aversion parameter $\lambda$. What is the relationship of variance and $\lambda$ in the limit as $\lambda \rightarrow 0$? See the paper by Busseti, E., Ryu, E. K., & Boyd, S. (2016) for ideas on how to perform this analysis.

# ## Bibliographic Notes
# 
# The Kelly Criterion has been included in many tutorial introductions to finance and probability, and the subject of popular accounts.
# 
# > Poundstone, W. (2010). Fortune's formula: The untold story of the scientific betting system that beat the casinos and Wall Street. Hill and Wang. https://www.onlinecasinoground.nl/wp-content/uploads/2020/10/Fortunes-Formula-boek-van-William-Poundstone-oa-Kelly-Criterion.pdf
# 
# > Thorp, E. O. (2017). A man for all markets: From Las Vegas to wall street, how i beat the dealer and the market. Random House.
# 
# > Thorp, E. O. (2008). The Kelly criterion in blackjack sports betting, and the stock market. In Handbook of asset and liability management (pp. 385-428). North-Holland. https://www.palmislandtraders.com/econ136/thorpe_kelly_crit.pdf
# 
# > MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010). Good and bad properties of the Kelly criterion. Risk, 20(2), 1.  https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
# 
# > MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2011). The Kelly capital growth investment criterion: Theory and practice (Vol. 3). world scientific. https://www.worldscientific.com/worldscibooks/10.1142/7598#t=aboutBook
# 
# >  https://en.wikipedia.org/wiki/Kelly_criterion
# 
# Luenberger's investment wheel is 
# 
# > Luenberger, D. (2009). Investment science: International edition. OUP Catalogue.  https://global.oup.com/ushe/product/investment-science-9780199740086
# 
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
# > Sun, Q., & Boyd, S. (2018). Distributional robust Kelly gambling. arXiv preprint arXiv: 1812.10371. https://web.stanford.edu/~boyd/papers/pdf/robust_kelly.pdf
# 
# The recent work by CH Hsieh extends these concepts in important ways for real-world implementation.
# 
# > Hsieh, C. H. (2022). On Solving Robust Log-Optimal Portfolio: A Supporting Hyperplane Approximation Approach. arXiv preprint arXiv:2202.03858. https://arxiv.org/pdf/2202.03858
# 

# In[ ]:




