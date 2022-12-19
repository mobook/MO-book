#!/usr/bin/env python
# coding: utf-8

# # Optimal Growth Portfolios
# 
# Among the reasons why Kelly was neglected by investors were high profile critiques by the most famous economist of the 20th Century, Paul Samuelson. Samuelson objected on several grounds, among them is a lack of risk aversion that results in large bets and risky short term behavior, and that Kelly's result is applicable to only one of many utility functions that describe investor preferences. The controversy didn't end there, however, as other academic economists, including Harry Markowitz, and practitioners found ways to adapt the Kelly criterion to investment funds.
# 
# This notebook presents solutions to Kelly's problem for optimal growth portfolios using exponential cones. A significant feature of this notebook is the the inclusion of a risk constraints recently proposed by Boyd and coworkers. These notes are based on recent papers such as Cajas (2021), Busseti, Ryu and Boyd (2016), Fu, Narasimhan, and Boyd (2017). Additional bibliographic notes are provided at the end of the notebook.

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_mosek()


# ## Portfolio Design for Optimal Growth

# ### Model
# 
# Given the models developed for the examples above, we now consider a set $N$ of financial assets trading in efficient markets. The historical record consists of a matrix $R \in \mathbb{R}^{T\times N}$ of gross returns where $T$ is the set of observations. The weights $w_n \geq 0$ for $n\in N$ denote the fraction of the portfolio invested in each asset $n$. Any portion of the portfolio not invested in traded assets is assumed to have a gross risk-free return $R_f = 1 + r_f$ where $r_f$ is the return on a risk-free asset.
# 
# Assuming the gross returns are independent and identically distributed random variates representative of future returns, the investment model becomes
# 
# $$
# \begin{align}
# \max_{w_n \geq 0}\quad & \frac{1}{T} \sum_{t\in T} \log(R_t) \\
# \text{s.t.}\quad \\
# & R_t = R_f + \sum_{n\in N} w_n (R_{t, n} - R_f) & \forall t\in T\\
# \end{align}
# $$
# 
# Note this formulation allows the sum of weights $\sum_{n\in N} w_n$ to be greater than one. In that case the investor would be investing more than the value of the portfolio in traded assets. In other words the investor would be creating a leveraged portfolio by borrowing money at a rate $R_f$. 
# 
# $$\sum_{n\in N} w_n \leq E_M$$
# 
# where $E_M$ is the "equity multiplier." A value $E_M \leq 1$ restricts the total investment to be less than or equal to the equity available to the investor. A value $E_M > 1$ allows the investor to leverage the available equity by borrowing money at a gross rate $R_f = 1 + r_f$. 
# 
# Using techniques demonstrated in the examples above, this model can be reformulated with exponential cones.
# 
# $$
# \begin{align}
# \max_{w_n}\quad & \frac{1}{T} \sum_{t\in T} q_t \\
# \text{s.t.}\quad \\
# & (R_f + \sum_{n\in N}w_n (R_{t,n} - R_f), 1, q_t) \in K_{exp} & \forall t \in T \\
# & \sum_{n\in N} w_n \leq E_M \\
# & w_n \geq 0  & \forall n\in N \\
# \end{align} 
# $$
# 
# For the risk constrained case, we consider a constraint
# 
# $$\mathbb{E}[R^{-\lambda}] \leq R_f^{-\lambda}$$
# 
# where $\lambda$ is a risk aversion parameter. Assuming the historical returns are equiprobable 
# 
# $$\frac{1}{T} \sum_{t\in T} R_t^{-\lambda} \leq R_f^{-\lambda}$$
# 
# The risk constraint is satisfied for any $w_n$ if the risk aversion parameter $\lambda=0$. For any value $\lambda > 0$ the risk constraint has a feasible solution $w_n=0$ for all $n \in N$. Recasting as a sum of exponentials,
# 
# $$\frac{1}{T} \sum_{t\in T} e^{- \lambda\log(R_t)} \leq R_f^{-\lambda}$$
# 
# Using the $q_t \leq \log(R_t)$ as used in the examples above, and $u_t \geq e^{- \lambda q_t}$, we get the risk constrained model optimal log growth. 
# 
# Given a risk-free rate of return $R_f$, a maximum equity multiplier $E_M$, and value $\lambda \geq 0$ for the risk aversion, risk constrained Kelly portfolio is given the solution to
# 
# $$
# \begin{align}
# \max_{w_n, q_t, u_t}\quad & \frac{1}{T} \sum_{t\in T} q_t \\
# \text{s.t.}\quad \\
# & \frac{1}{T} \sum_{t\in T} u_t \leq R_f^{-\lambda} \\
# & (u_t, 1, \lambda q_t) \in K_{exp} & \forall t\in T \\
# & (R_f + \sum_{n\in N}w_n (R_{t,n} - R_f), 1, q_t) \in K_{exp} & \forall t \in T \\
# & \sum_{n\in N} w_n \leq E_M \\
# & w_n \geq 0 & \forall n \in N \\
# \end{align}
# $$
# 
# The following cells demonstrate an implementation of the model using the Pyomo kernel library and Mosek solver.

# ### Data set
# 
# We begin by reading recennt historical prices for a selected set of trading symbols using [yfinance](https://github.com/ranaroussi/yfinance).

# In[ ]:


get_ipython().system('pip install yfinance --upgrade -q')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import datetime
import yfinance as yf

symbols = {
    '^GSPC': "S&P 500 Index",
    'AAPL': "Apple",
    'MCD': "McDonald's",
    'MMM': "3 M",
    'MSFT': "Microsoft",
    'XOM': "ExxonMobil",
}

# end date is today
end = datetime.datetime.today().date()

# start date is n_years earlier
n_years = 2
start = end - datetime.timedelta(int(n_years*365))

# get stock price data
data = [yf.download(symbol, start=start, end=end)["Adj Close"] for symbol in symbols]
S = pd.concat(data, axis=1)
S.columns = [symbol for symbol in symbols]

# get gross returns
R = S/S.shift(1)
R.dropna(inplace=True)


# In[ ]:


# plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
S.divide(S.iloc[0]/100).plot(ax=ax[0], grid=True, title="Normalized Prices")
R.plot(ax=ax[1], grid=True, title="Gross Returns", alpha=0.5)
fig.tight_layout()


# ### Pyomo Implementation
# 
# The Pyomo implementation for the risk-constrained Kelly portfolio accepts three parameters, the risk-free gross returns $R_f$, the maximum equity multiplier, and the risk-aversion parameter.

# In[ ]:


def kelly_portfolio(R, Rf=1, EM=1, lambd=0):

    m = pmo.block()
    
    # return parameters with the model
    m.Rf = Rf
    m.EM = EM
    m.lambd = lambd
    
    # index lists
    m.T = R.index
    m.N = R.columns
    
    # decision variables
    m.q = pmo.variable_dict({t: pmo.variable() for t in m.T})
    m.w = pmo.variable_dict({n: pmo.variable(lb=0) for n in m.N})

    # objective
    m.ElogR = pmo.objective(sum(m.q[t] for t in m.T)/len(m.T), sense=pmo.maximize)

    # conic constraints on return
    m.R = pmo.expression_dict({t: pmo.expression(Rf + sum(m.w[n]*(R.loc[t, n] - Rf) for n in m.N)) for t in m.T})
    m.c = pmo.block_dict({t:pmo.conic.primal_exponential.as_domain(m.R[t], 1, m.q[t]) for t in m.T})

    # risk constraints
    m.u = pmo.variable_dict({t: pmo.variable() for t in m.T})
    m.u_sum = pmo.constraint(sum(m.u[t] for t in m.T)/len(m.T) <= Rf**(-lambd))
    m.r = pmo.block_dict({t:pmo.conic.primal_exponential.as_domain(m.u[t], 1, -lambd*m.q[t]) for t in m.T})
    
    # equity multiplier constraint
    m.w_sum = pmo.constraint(sum(m.w[n] for n in m.N) <= EM)

    pmo.SolverFactory('mosek_direct').solve(m)
    
    return m

def kelly_report(m):
    s = ""
    #print report
    s += f"\nRisk Free Return = {100*(np.exp(252*np.log(m.Rf)) - 1):0.2f} %"
    s += f"\nEquity Multiplier Limit = {m.EM:0.5f}"
    s += f"\nRisk Aversion = {m.lambd:0.5f}"
    s += "\n\nPortfolio"
    s += "\n"
    s += "\n".join([f"{n:8s} {symbols[n]:30s}  {100*m.w[n]():8.2f} %" for n in m.N])
    s += f"\n{'':8s} {'Risk Free':30s}  {100*(1 - sum(m.w[n]() for n in R.columns)):8.2f} %"
    s += f"\n\nAnnualized return = {100*(np.exp(252*m.ElogR()) - 1):0.2f} %"
    print(s)
    
    df = pd.DataFrame(pd.Series([m.R[t]() for t in m.T]), columns=["Kelly Portfolio"])
    df.index = m.T
    
    fix, ax = plt.subplots(1, 1, figsize=(8, 3))
    S.divide(S.iloc[0]/100).plot(ax=ax, grid=True, title="Normalized Prices", alpha=0.6, lw=0.5)
    df.cumprod().multiply(100).plot(ax=ax, lw=3, grid=True)
    ax.legend([symbols[n] for n in m.N] + ["Kelly Portfolio"], bbox_to_anchor=(1.05, 1.05))

   
# parameter values
Rf = Rf=np.exp(np.log(1.05)/252)
EM = 2
lambd = 10

m = kelly_portfolio(R, Rf, EM, lambd)
kelly_report(m)


# ### Effects of the Risk-Aversion Parameter

# In[ ]:


lambd = 10**np.linspace(0, 3)

results = [kelly_portfolio(R, Rf=1, EM=1, lambd=_) for _ in lambd]


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

ax[0].semilogx([m.lambd for m in results], [100*(np.exp(252*m.ElogR()) - 1) for m in results])
ax[0].set_title('Portfolio Return vs Risk Aversion')
ax[0].set_ylabel("annual %")
ax[0].grid(True)

ax[1].semilogx([m.lambd for m in results], [[m.w[n]() for n in R.columns] for m in results])
ax[1].set_ylabel('weights')
ax[1].set_xlabel('risk aversion $\lambda$')
ax[1].legend([symbols[n] for n in m.N], bbox_to_anchor=(1.05, 1.05))
ax[1].grid(True)
ax[1].set_ylim(0, EM)

fig.tight_layout()


# ### Effects of the Equity Multiplier Parameter

# In[ ]:


EM = np.linspace(0.0, 2.0)

results = [kelly_portfolio(R, Rf=1, EM=_, lambd=10) for _ in EM]


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

ax[0].plot([m.EM for m in results], [100*(np.exp(252*m.ElogR()) - 1) for m in results])
ax[0].set_title('Portfolio Return vs Equity Multiplier')
ax[0].set_ylabel("annual return %")
ax[0].grid(True)

ax[1].plot([m.EM for m in results], [[m.w[n]() for n in R.columns] for m in results])
ax[1].set_ylabel('weights')
ax[1].set_xlabel('Equity Multiplier')
ax[1].legend([symbols[n] for n in m.N], bbox_to_anchor=(1.05, 1.05))
ax[1].grid(True)
ax[1].set_ylim(0, )

fig.tight_layout()


# ### Effect of Risk-free Interest Rate

# In[ ]:


Rf = np.exp(np.log(1 + np.linspace(0, 0.20))/252)

results = [kelly_portfolio(R, Rf=_, EM=1, lambd=10) for _ in Rf]


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

Rf = np.exp(252*np.log(np.array([_.Rf for _ in results])))
ax[0].plot(Rf, [100*(np.exp(252*m.ElogR()) - 1) for m in results])
ax[0].set_title('Portfolio Return vs Risk-free Rate')
ax[0].set_ylabel("annual return %")
ax[0].grid(True)

ax[1].plot(Rf, [[m.w[n]() for n in R.columns] for m in results])
ax[1].set_ylabel('weights')
ax[1].set_xlabel('Risk-free Rate')
ax[1].legend([symbols[n] for n in m.N], bbox_to_anchor=(1.05, 1.05))
ax[1].grid(True)
ax[1].set_ylim(0, )

fig.tight_layout()


# ## Extensions
# 
# 1. The examples cited in this notebook assume knowledge of the probability mass distribution. Recent work by Sun and Boyd (2018) and Hsieh (2022) suggest models for finding investment strategies for cases where the distributions are not perfectly known. They call the "distributional robust Kelly gambling." A useful extension to this notebook would be to demonstrate a robust solution to one or more of the examples.

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




