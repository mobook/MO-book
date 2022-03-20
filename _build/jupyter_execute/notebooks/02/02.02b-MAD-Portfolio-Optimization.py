#!/usr/bin/env python
# coding: utf-8

# # MAD Portfolio Optimization
# 
# NOTE: THIS IS IN EARLY DEVELOPMENT.  STILL NEED TO FINALIZE MODEL, TEST, AND REVISE NARRATIVE.  PRIORITY IS TO FINISH BY 3/18.

# In[1]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# In[2]:


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as stats

import pyomo.environ as pyo


# ## Investment objectives
# 
# * Maximize returns
# * Reduce Risk through diversification

# What we observe is that even a small amount of diversification can dramatically reduce the downside risk of experiencing a loss. We also see the upside potential has been reduced. What hasn't changed is the that average profit remains at \$50,000. Whether or not the loss of upside potential in order to reduce downside risk is an acceptable tradeoff depends on your individual attitude towards risk. 

# ## Read historical asset prices
# 
# READ DATA PREVIOUSLY IMPORTED.

# In[3]:


# read historical asset prices

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

data_path = os.path.join("data", "stocks")
files = glob.glob(os.path.join(data_path, "*.csv"))

assets = pd.DataFrame()
for filename in sorted(files):
    data = pd.read_csv(filename, index_col=0)
    sym = filename.split("/")[-1].split(".")[0]
    assets[sym] = data["Adj Close"]
    
assets.fillna(method="bfill", inplace=True)
assets.fillna(method="ffill", inplace=True)
    
assets.plot(logy=True, figsize=(12, 8), grid=True, lw=1, title="Adjusted Close")
plt.legend(bbox_to_anchor=(1.0, 1.0))


# In[4]:


# scaled asset prices

assets_scaled = assets.div(assets.iloc[0])
assets_scaled.plot(figsize=(12, 8), grid=True, lw=1, title="Adjusted Close: Scaled")
plt.legend(bbox_to_anchor=(1.0, 1.0))


# In[5]:


# daily returns

daily_returns = assets.diff()[1:]/assets.shift(1)[1:]

fig, ax = plt.subplots(6, 5, figsize=(12, 10), sharex=True, sharey=True)
ax = ax.flatten()

for a, s in zip(ax.flatten(), sorted(daily_returns.columns)):
    daily_returns[s].plot(ax=a, lw=1, title=s, grid=True)
    
plt.tight_layout()


# In[6]:


# distributions of returns

daily_returns = assets.diff()[1:]/assets.shift(1)[1:]

fig, ax = plt.subplots(6, 5, figsize=(12, 10), sharex=True, sharey=True)
ax = ax.flatten()

for a, s in zip(ax.flatten(), daily_returns.columns):
    daily_returns[s].hist(ax=a, lw=1, grid=True, bins=50)
    mean_return = daily_returns[s].mean()
    mean_absolute_deviation = abs((daily_returns[s] - mean_return)).mean()
    a.set_title(f"{s} = {mean_return:0.5f}")
    a.set_xlim(-0.08, 0.08)
    a.axvline(mean_return, color='r', linestyle="--")
    a.axvline(mean_return + mean_absolute_deviation, color='g', linestyle='--')
    a.axvline(mean_return - mean_absolute_deviation, color='g', linestyle='--')
    
plt.tight_layout()


# In[7]:


# bar charts of mean return and mean absolute deviation in returns

daily_returns = assets.diff()[1:]/assets.shift(1)[1:]
mean_return = daily_returns.mean()
mean_absolute_deviation = abs(daily_returns - mean_return).mean()

fig, ax = plt.subplots(1, 2, figsize = (12, 0.35*len(daily_returns.columns)))
mean_return.plot(kind='barh', ax=ax[0], title="Mean Return")
ax[0].invert_yaxis()
mean_absolute_deviation.plot(kind='barh', ax=ax[1], title='Mean Absolute Deviation');
ax[1].invert_yaxis()


# In[8]:


# plot return vs risk

daily_returns = assets.diff()[1:]/assets.shift(1)[1:]
mean_return = daily_returns.mean()
mean_absolute_deviation = abs(daily_returns - mean_return).mean()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
for s in assets.keys():
    ax.plot(mean_absolute_deviation[s], mean_return[s], 's', ms=8)
    ax.text(mean_absolute_deviation[s]*1.03, mean_return[s], s)

ax.set_xlim(0, 1.1*max(mad))
ax.axhline(0, color='r', linestyle='--')
ax.set_title('Return vs Risk')
ax.set_xlabel('Mean Absolute Deviation in Daily Returns')
ax.set_ylabel('Mean Daily Return')
ax.grid(True)


# ## Porfolio
# 
# The return on a portfolio with weights $w_a$ for asset $a$ is 
# 
# 
# $$
# \begin{align*}
# MAD  = \min \frac{1}{T} \sum_{t\in TIME} \left| \sum_{a\in ASSETS} w_a (r_{t, a} - \bar{r}_a) \right|
# \end{align*}
# $$
# 
# where $r_{t, a}$ is the return on asset $a$ at time $t$, $\bar{r}_a$ is the mean return for asset $a$, and $w_a$ is the fraction of the total portfolio that is invested in asset $a$.

# In[15]:


import pyomo.environ as pyo

def mad_portfolio(assets):
    
    daily_returns = assets.diff()[1:]/assets.shift(1)[1:]
    mean_return = daily_returns.mean()

    m = pyo.ConcreteModel()
    
    m.Rp = pyo.Param(mutable=True, default=0)
    m.W_lb = pyo.Param(mutable=True, default=-10)
    
    m.ASSETS = pyo.Set(initialize=assets.columns)
    m.TIME = pyo.RangeSet(len(daily_returns.index))
    
    m.w = pyo.Var(m.ASSETS, bounds=(m.W_lb, 10))
    m.r_pos = pyo.Var(m.TIME, domain=pyo.NonNegativeReals)
    m.r_neg = pyo.Var(m.TIME, domain=pyo.NonNegativeReals)
    
    @m.Constraint(m.TIME)
    def portfolio_returns(m, t):
        date = daily_returns.index[t-1]
        return m.r_pos[t] - m.r_neg[t] == sum(m.w[a]*(daily_returns.loc[date, a] - mean_return[a]) for a in m.ASSETS)
    
    @m.Objective(sense=pyo.minimize)
    def mad(m):
        return sum(m.r_pos[t] + m.r_neg[t] for t in m.TIME)/len(m.TIME)
    
    @m.Constraint()
    def sum_of_weights(m):
        return 1 == sum(m.w[a] for a in m.ASSETS)
    
    @m.Constraint()
    def mean_portfolio_return(m):
        return sum(m.w[a] * mean_return[a] for a in m.ASSETS) == m.Rp
    
    return m
    
m = mad_portfolio(assets)
pyo.SolverFactory('cbc').solve(m)

daily_returns = assets.diff()[1:]/assets.shift(1)[1:]
mean_return = daily_returns.mean()
mean_absolute_deviation = abs(daily_returns - mean_return).mean()
mad_portfolio_weights = pd.DataFrame([m.w[a]() for a in sorted(m.ASSETS)], index=sorted(m.ASSETS))

fig, ax = plt.subplots(1, 3, figsize = (12, 0.35*len(daily_returns.columns)))
mad_portfolio_weights.plot(kind='barh', ax=ax[0], title="MAD Portfolio Weights")
ax[0].invert_yaxis()
mean_return.plot(kind='barh', ax=ax[1], title="Mean Return")
ax[1].invert_yaxis()
mean_absolute_deviation.plot(kind='barh', ax=ax[2], title='Mean Absolute Deviation');
ax[2].invert_yaxis()


# In[18]:


# plot return vs risk

daily_returns = assets.diff()[1:]/assets.shift(1)[1:]
mean_return = daily_returns.mean()
mean_absolute_deviation = abs(daily_returns - mean_return).mean()

fig, ax = plt.subplots(1, 1, figsize=(10,6))
for s in assets.keys():
    ax.plot(mean_absolute_deviation[s], mean_return[s], 's', ms=8)
    ax.text(mean_absolute_deviation[s]*1.03, mean_return[s], s)
    
ax.set_xlim(0, 1.1*max(mean_absolute_deviation))
ax.axhline(0, color='r', linestyle='--')
ax.set_title('Return vs Risk')
ax.set_xlabel('Mean Absolute Deviation in Daily Returns')
ax.set_ylabel('Mean Daily Return')
ax.grid(True)

import numpy as np

m.W_lb = -100
for Rp in np.linspace(0, 0.0020, 20):
    m.Rp = Rp
    pyo.SolverFactory('cbc').solve(m)
    mad_portfolio_weights = pd.DataFrame([m.w[a]() for a in sorted(m.ASSETS)], index=sorted(m.ASSETS))

    portfolio_returns = daily_returns.dot(mad_portfolio_weights)
    portfolio_mean_return = portfolio_returns.mean()
    portfolio_mean_absolute_deviation = abs(portfolio_returns - portfolio_mean_return).mean()
    ax.plot(portfolio_mean_absolute_deviation, portfolio_mean_return, 'ro', ms=10)


# ## Statistics of daily asset returns

# In[14]:


S_hist = pd.read_csv('data/Historical_Adjusted_Close.csv', index_col=0)

S_hist.dropna(axis=1, how='any', inplace=True)
S_hist.index = pd.DatetimeIndex(S_hist.index)

portfolio = list(S_hist.columns)
print(portfolio)
S_hist.tail()


# ## Select a recent subperiod of the historical data

# In[43]:


nYears = 1.5
start = S_hist.index[-int(nYears*252)]
end = S_hist.index[-1]

print('Start Date:', start)
print('  End Date:', end)

S = S_hist.loc[start:end]
S.head()


# In[44]:


fig, ax = plt.subplots(figsize=(14,9))
S.plot(ax=ax, logy=True)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)


# ## Return on a portfolio
# 
# Given a portfolio with value $W_t$ at time $t$, return on the portfolio at $t_{t +\delta t}$ is defined as
# 
# \begin{align*}
# r_{t + \delta t} & = \frac{W_{t + \delta t} - W_{t}}{W_{t}}
# \end{align*}
# 
# For the period from $[t, t+\delta t)$ we assume there are $n_{j,t}$ shares of asset $j$ with a starting value of $S_{j,t}$ per share. The initial and final values of the portfolio are then 
# 
# \begin{align*}
# W_t & = \sum_{j=1}^J n_{j,t}S_{j,t} \\
# W_{t+\delta t} & = \sum_{j=1}^J n_{j,t}S_{j,t + \delta t}
# \end{align*}
# 
# The return of the portfolio is given by
# 
# \begin{align*}
# r_{t+\delta t} & = \frac{W_{t + \delta t} - W_{t}}{W_{t}} \\
# & = \frac{\sum_{j=1}^Jn_{j,t}S_{j,t+\delta t} - \sum_{j=1}^J n_{j,t}S_{j,t}}{W_{t}} \\
# & = \frac{\sum_{j=1}^J n_{j,t}S_{j,t}r_{j, t+\delta t}}{W_{t}} \\
# & = \sum_{j=1}^J \frac{n_{j,t}S_{j,t}}{W_{t}} r_{j, t+\delta t}
# \end{align*}
# 
# where $r_{j,t+\delta t}$ is the return on asset $j$ at time $t+\delta t$. 
# 
# Defining $W_{j,t} = n_{j,t}S_{j,t}$ as the wealth invested in asset $j$ at time $t$, then $w_{j,t} = n_{j,t}S_{j,t}/W_{t}$ is the fraction of total wealth invested in asset $j$ at time $t$. The portfolio return is then given by 
# 
# \begin{align*}
# r_{t+\delta t} & = \sum_{j=1}^J w_{j,t} r_{j, t+\delta t} 
# \end{align*}
# 
# on a single interval extending from $t$ to $t + \delta t$.

# ### Equally weighted portfolio
# 
# An equally weighted portfolio allocates an equal amount of capital to each component of the portfolio. The allocation can be done once and held fixed thereafter, or could be reallocated periodically as asset prices change in relation to one another.

# #### Constant fixed allocation
# 
# If the initial allocation among $J$ assets takes place at $t=0$, then
# 
# $$w_{j,0} = \frac{1}{J} = \frac{n_{j,0} S_{j, t=0}}{W_{0}}$$
# 
# The number of assets of type $j$ included in the portfolio is given by
# 
# $$n_{j,0} = \frac{W_0}{J S_{j,0}} $$
# 
# which is then fixed for all later times $t > 0$. The value of the portfolio is given by
# 
# \begin{align*}
# W_t & = \sum_{j=1}^J n_{j,0}S_{j,t} \\
# & = \frac{W_{0}}{J} \sum_{j=1}^J \frac{S_{j,t}}{S_{j,0}}
# \end{align*}
# 
# Note that this portfolio is guaranteed to be equally weighted only at $t=0$. Changes in the relative prices of assets cause the relative weights of assets in the portfolio to change over time.

# #### Continually rebalanced
# 
# Maitaining an equally weighted portfolio requires buying and selling of component assests as prices change relative to each other. To maintain an equally portfolio comprised of $J$ assets where the weights are constant in time,
# 
# \begin{align*}
# w_{j,t} & = \frac{1}{J} = \frac{n_{j,t}S_{j,t}}{W_t} & \forall j, t
# \end{align*}
# 
# Assuming the rebalancing occurs at fixed points in time $t_k$ separated by time steps $\delta t$, then on each half-closed interval $[t_k, t_k+\delta t)$
# 
# \begin{align*}
# n_{j,t} & = \frac{W_{t_k}}{J S_{j,t_k}} \\
# \end{align*}
# 
# The portfolio
# 
# \begin{align*}
# W_{t_k + \delta t} & = \sum_{j=1}^J n_{j,t_k} S_{j, t_k + \delta t}
# \end{align*}
# 
# \begin{align*}
# W_{t_k + \delta t} & = W_{t_k} \sum_{j=1}^J  \frac{S_{j, t_k + \delta t}}{J S_{j,t_k}}
# \end{align*}
# 
# Letting $t_{k+1} = t_k + \delta t$, then the following recursion describes the dynamics of an equally weighted,  continually rebalanced portfolio at the time steps $t_0, t_1, \ldots$. Starting with values $W_{t_0}$ and $S_{j, t_0}$, 
# 
# \begin{align*}
# n_{j,t_k} & = \frac{W_{t_k}}{J S_{j,t_k}} \\
# W_{t_{k+1}} & = \sum_{j=1}^J  n_{j,t_k} S_{j, t_{k+1}}
# \end{align*}
# 
# which can be simulated as a single equation
# 
# \begin{align*}
# W_{t_{k+1}} & = W_{t_k} \sum_{j=1}^J  \frac{S_{j, t_{k+1}}}{J S_{j,t_k}}
# \end{align*}
# 
# or in closed-form
# 
# \begin{align*}
# W_{t_{K}} & = W_{0} \prod_{k=0}^{K-1} \sum_{j=1}^J  \frac{S_{j, t_{k+1}}}{J S_{j,t_k}}
# \end{align*}

# In[45]:


plt.figure(figsize=(12,6))

portfolio = S.columns
J = len(portfolio)

# equal weight with no rebalancing
n = 100.0/S.iloc[0]/J
W_fixed = sum(n[s]*S[s] for s in portfolio)
W_fixed.plot(color='r',lw=4)

# equal weighting with continual rebalancing
R = (S[1:]/S.shift(1)[1:]).sum(axis=1)/len(portfolio)
W_rebal = 100*R.cumprod()
W_rebal.plot(color='b', lw=4)

# individual assets
for s in portfolio:
    (100.0*S[s]/S[s][0]).plot(lw=0.4)
    
plt.legend(['Fixed Allocation','Continually Rebalanced'])
plt.ylabel('Value');
plt.title('Value of an equally weighted portfolio')
plt.grid(True)


# In[49]:


plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
W_fixed.plot()
W_rebal.plot()
plt.legend(['Fixed Allocation','Continually Rebalanced'])
plt.ylabel('Value')
plt.title('Comparing a Fixed and Continually Rebalanced Portfolio')
plt.grid(True)

plt.subplot(2,1,2)
(100.0*(W_rebal-W_fixed)/W_fixed).plot()
plt.title('Added value of a Rebalanced Portfolio relative to a Fixed Portfolio')
plt.ylabel('percent')
plt.grid(True)

plt.tight_layout()


# ### Component returns
# 
# Given data on the prices for a set of assets over an historical period $t_0, t_1, \ldots, t_K$, an estimate the mean arithmetic return is given by the mean value
# 
# \begin{align*}
# \hat{r}_{j,t_K} & = \frac{1}{K}\sum_{k=1}^{K} r_{t_k} \\
# & = \sum_{k=1}^{K} \frac{S_{j,t_{k}}-S_{j,t_{k-1}}}{S_{j,t_{k-1}}}
# \end{align*}
# 
# At any point in time, $t_k$, a mean return can be computed using the previous $H$ intervals
# 
# \begin{align*}
# \hat{r}^H_{j,t_k} & = \frac{1}{H}\sum_{h=0}^{H-1} r_{t_{k-h}} \\
# & = \frac{1}{H} \sum_{h=0}^{H-1}\frac{S_{j,t_{k-h}} - S_{j,t_{k-h-1}}}{S_{j,t_{k-h-1}}}
# \end{align*}
# 
# Arithmetic returns are computed so that subsequent calculations combine returns across components of a portfolio.

# ## Measuring deviation in component returns

# ### Mean absolute deviation

# In[30]:


def roll(H):
    """Plot mean returns, mean absolute deviation, and standard deviation for last H days."""
    K = len(S.index)
    R = S[K-H-1:K].diff()[1:]/S[K-H-1:K].shift(1)[1:]
    AD = abs(R - R.mean())
    
    plt.figure(figsize = (12, 0.35*len(R.columns)))
    ax = [plt.subplot(1,3,i+1) for i in range(0,3)]
    
    idx = R.columns.argsort()[::-1]

    R.mean().iloc[idx].plot(ax=ax[0], kind='barh')
    ax[0].set_title('Mean Returns');
    
    AD.mean().iloc[idx].plot(ax=ax[1], kind='barh')
    ax[1].set_title('Mean Absolute Difference')

    R.std().iloc[idx].plot(ax=ax[2], kind='barh')
    ax[2].set_title('Standard Deviation')
    
    for a in ax: a.grid(True)
    plt.tight_layout()

roll(500)


# ### Comparing mean absolute deviation to standard deviation

# In[51]:


R = (S.diff()[1:]/S.shift(1)[1:]).dropna(axis=0, how='all')
AD = abs(R - R.mean())

plt.plot(R.std(), AD.mean(), 'o')
plt.xlabel('Standard Deviation')
plt.ylabel('Mean Absolute Deviation')

plt.plot([0,R.std().max()],[0,np.sqrt(2.0/np.pi)*R.std().max()])
plt.legend(['Portfolio Components','sqrt(2/np.pi)'],loc='best')
plt.grid(True)


# ### Return versus mean absolute deviation for an equally weighted continually rebalanced portfolio

# In[52]:


plt.figure(figsize=(10,6))
for s in portfolio:
    plt.plot(AD[s].mean(), R[s].mean(),'s')
    plt.text(AD[s].mean()*1.03, R[s].mean(), s)
    
R_equal = W_rebal.diff()[1:]/W_rebal[1:]
M_equal = abs(R_equal-R_equal.mean()).mean()

plt.plot(M_equal, R_equal.mean(), 'ro', ms=15)

plt.xlim(0, 1.1*max(AD.mean()))
plt.ylim(min(0, 1.1*min(R.mean())), 1.1*max(R.mean()))
plt.plot(plt.xlim(),[0,0],'r--');
plt.title('Risk/Return for an Equally Weighted Portfolio')
plt.xlabel('Mean Absolute Deviation')
plt.ylabel('Mean Daily Return');
plt.grid(True)


# ## MAD porfolio

# The linear program is formulated and solved using Pyomo. 

# In[15]:


R.head()


# The decision variables will be indexed by date/time.  The pandas dataframes containing the returns data are indexed by timestamps that include characters that cannot be used by the GLPK solver. Therefore we create a dictionary to translate the pandas timestamps to strings that can be read as members of a GLPK set. The strings denote seconds in the current epoch as defined by python.

# In[16]:


a = R - R.mean()
a.head()


# In[ ]:





# ## Minimizing MAD for a portfolio

# In[17]:


from pyomo.environ import *

a = R - R.mean()

m = ConcreteModel()

m.w = Var(R.columns, domain=NonNegativeReals)
m.y = Var(R.index, domain=NonNegativeReals)

m.MAD = Objective(expr=sum(m.y[t] for t in R.index)/len(R.index), sense=minimize)

m.c1 = Constraint(R.index, rule = lambda m, t: m.y[t] + sum(a.loc[t,s]*m.w[s] for s in R.columns) >= 0)
m.c2 = Constraint(R.index, rule = lambda m, t: m.y[t] - sum(a.loc[t,s]*m.w[s] for s in R.columns) >= 0)
m.c3 = Constraint(expr=sum(R[s].mean()*m.w[s] for s in R.columns) >= R_equal.mean())
m.c4 = Constraint(expr=sum(m.w[s] for s in R.columns)==1)

SolverFactory('glpk').solve(m)

w = {s: m.w[s]() for s in R.columns}

plt.figure(figsize = (15,0.35*len(R.columns)))

plt.subplot(1,3,1)
pd.Series(w).plot(kind='barh')
plt.title('Porfolio Weight');

plt.subplot(1,3,2)
R.mean().plot(kind='barh')
plt.title('Mean Returns');

plt.subplot(1,3,3)
AD.mean().plot(kind='barh')
plt.title('Mean Absolute Difference');


# In[54]:


P_mad = pd.Series(0, index=S.index)
for s in portfolio:
    P_mad += 100.0*w[s]*S[s]/S[s][0]
    
plt.figure(figsize=(12,6))
P_mad.plot()
W_rebal.plot()
plt.legend(['MAD','Equal'],loc='best')
plt.ylabel('Unit Value')
plt.grid(True)


# In[19]:


plt.figure(figsize=(10,6))
for s in portfolio:
    plt.plot(AD[s].mean(), R[s].mean(),'s')
    plt.text(AD[s].mean()*1.03, R[s].mean(), s)
    
#R_equal = P_equal.diff()[1:]/P_equal[1:]
R_equal = np.log(W_rebal/W_rebal.shift(+1))
M_equal = abs(R_equal-R_equal.mean()).mean()

plt.plot(M_equal, R_equal.mean(), 'ro', ms=15)

#R_mad = P_mad.diff()[1:]/P_mad[1:]
R_mad = np.log(P_mad/P_mad.shift(+1))
M_mad = abs(R_mad-R_mad.mean()).mean()

plt.plot(M_mad, R_mad.mean(), 'go', ms=15)

for s in portfolio:
    if w[s] >= 0.0001:
        plt.plot([M_mad, AD[s].mean()],[R_mad.mean(), R[s].mean()],'g--')
    if w[s] <= -0.0001:
        plt.plot([M_mad, AD[s].mean()],[R_mad.mean(), R[s].mean()],'r--')

plt.xlim(0, 1.1*max(AD.mean()))
plt.ylim(min(0, 1.1*min(R.mean())), 1.1*max(R.mean()))
plt.plot(plt.xlim(),[0,0],'r--');
plt.title('Risk/Return for an Equally Weighted Portfolio')
plt.xlabel('Mean Absolute Deviation')
plt.ylabel('Mean Daily Return')


# In[53]:


import pulp

# mean absolute deviation for the portfolio
m = pulp.LpVariable('m', lowBound = 0)

# dictionary of portfolio weights
w = pulp.LpVariable.dicts('w', portfolio, lowBound = 0)

# dictionary of absolute deviations of portfolio returns
y = pulp.LpVariable.dicts('y', t.values(), lowBound = 0)
z = pulp.LpVariable.dicts('z', t.values(), lowBound = 0)

# create problem instance
lp = pulp.LpProblem('MAD Portfolio',pulp.LpMinimize)

# add objective
lp += m

# calculate mean absolute deviation of portfolio returns
lp += m == pulp.lpSum([(y[k] + z[k]) for k in t.values()])/float(len(t))

# relate the absolute deviations to deviations in the portfolio returns
for ts in returns.index:
    lp += y[t[ts]] - z[t[ts]] == pulp.lpSum([w[s]*(returns[s][ts]-returns[s].mean()) for s in portfolio]) 
    
# portfolio weights
lp += pulp.lpSum([w[s] for s in portfolio]) == 1.0

# bound on average portfolio return
lp += pulp.lpSum([w[s]*(returns[s].mean()) for s in portfolio]) >= 0*R_equal.mean()

lp.solve()
print(pulp.LpStatus[lp.status])


# In[ ]:


figure(figsize = (15,0.35*len(returns.columns)))

ws = pd.Series({s: w[s].varValue for s in portfolio},index=portfolio)

subplot(1,3,1)
ws.plot(kind='barh')
title('Porfolio Weight');

subplot(1,3,2)
returns.mean().plot(kind='barh')
title('Mean Returns');

subplot(1,3,3)
abs(returns-returns.mean()).mean().plot(kind='barh')
title('Mean Absolute Difference');


# In[ ]:


P_mad = pd.Series(0,index=adjclose.index)
for s in portfolio:
    P_mad += 100.0*ws[s]*adjclose[s]/adjclose[s][0]
    
figure(figsize=(12,6))
P_mad.plot()
P_equal.plot()
legend(['MAD','Equal'],loc='best')
ylabel('Unit Value')


# In[ ]:


figure(figsize=(10,6))
for s in portfolio:
    plot(mad[s],rmean[s],'s')
    text(mad[s]*1.03,rmean[s],s)
    
axis([0, 1.1*max(mad), min([0,min(rmean)-.1*(max(rmean)-min(rmean))]), 1.1*max(rmean)])
ax = axis()
plot([ax[0],ax[1]],[0,0],'r--');

#R_equal = P_equal.diff()[1:]/P_equal[1:]
R_equal = log(P_equal/P_equal.shift(+1))
M_equal = abs(R_equal-R_equal.mean()).mean()

plot(M_equal,R_equal.mean(),'ro',ms=15)

#R_mad = P_mad.diff()[1:]/P_mad[1:]
R_mad = log(P_mad/P_mad.shift(+1))
M_mad = abs(R_mad-R_mad.mean()).mean()

for s in portfolio:
    if ws[s] >= 0.0001:
        plot([M_mad,mad[s]],[R_mad.mean(),rmean[s]],'g--')

plot(M_mad,R_mad.mean(),'go',ms=15)

title('Risk/Return for an Equally Weighted Portfolio')
xlabel('Mean Absolute Deviation')
ylabel('Mean Return')

grid();

