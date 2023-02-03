#!/usr/bin/env python
# coding: utf-8

# ```{index} single: application; portfolio
# ```
# ```{index} single: application; investment
# ```
# ```{index} single: solver; cplex
# ```
# ```{index} chance constraints
# ```
# # Markowitz portfolio optimization with chance constraints

# In[29]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cplex()


# In[30]:


from IPython.display import Markdown, HTML
import pyomo.environ as pyo
import numpy as np


# We consider here another variant of the Markowitz portfolio optimization problem, which we already encountered in the context of convex optimization [here](../05/markowitz_portfolio.ipynb) and in the context of conic optimization [here](../06/markowitz_portfolio_revisited.ipynb).
# 
# Assuming there is an initial unit capital $C$ that needs to be invested in a selection of $n$ possible assets, each of them with a unknown return rate $r_i$, $i=1,\dots,n$. Let $x$ be the vector whose $i$-th component $x_i$ describes the fraction of the capital invested in asset $i$. The return rate vector $r$ can be modelled by a multivariate Gaussian distribution with mean $\mu$ and covariance $\Sigma$. Assume there is also a risk-free asset with guaranteed return rate $R$ and let $\tilde{x}$ the amount invested in that asset. We want to determine the portfolio that maximizes the _expected_ return $\mathbb{E} ( R \tilde{x} + r^\top x )$, which in view of our assumptions rewrites as $ \mathbb{E} ( R \tilde{x} + r^\top x ) = R \tilde{x} + \mu^\top x$.
# 
# Additionally, we includ a _loss risk chance constraint_ of the form 
# 
# $$
# \mathbb{P} ( r^\top x \leq \alpha) \leq \beta.
# $$ 
# 
# Thanks to the assumption that $r$ is multivariate Gaussian, this chance constraint can be equivalently rewritten as
# 
# $$
#     \mu^\top x \geq \Phi^{-1}(1-\beta) \| \Sigma^{1/2} r \|_2 + \alpha,
# $$
# 
# which the theory guarantees to be a convex constraint if $\beta \leq 1/2$. The resulting portfolio optimization problem written as a SOCP is
# 
# $$
# \begin{align*}
#     \max \; & R \tilde{x} + \mu^\top x\\
#     \quad \text{ s.t. } & \Phi^{-1}(1-\beta) \| \Sigma^{1/2} x \|_2 \leq \mu^\top x - \alpha,\\
#     & \sum_{i=1}^n x_i + \tilde{x} = C, \\
#     & \tilde{x} \geq 0 \\
#     &  x_i \geq 0 & i=1,\dots,n.
# \end{align*}
# $$
# 
# We now implement as a Pyomo model and solve it for $n=3$, $\alpha = 0.6$, $\beta =0.3$, the given vector $\mu$ and semi-definite positive covariance matrix $\Sigma$.

# In[28]:


# we import the inverse CDF or quantile function for the standard normal norm.ppf() from scipy.stats
from scipy.stats import norm

# We set our risk threshold and risk levels (sometimes you may get an infeasible problem if the chance
# constraint becomes too tight!)
alpha = 0.6
beta = 0.3

# We specify the initial capital, the risk-free return the number of risky assets, their expected returns, and their covariance matrix. 
C = 1
R = 1.05
n = 3
mu = np.array([1.25, 1.15, 1.35])
Sigma = np.array([[1.5, 0.5, 2], [0.5, 2, 0], [2, 0, 5]])

# Check how dramatically the optimal solution changes if we assume i.i.d. deviations for the returns.
# Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# If you want to change covariance matrix, make sure you input a semi-definite positive one.
# The easiest way to generate a random covariance matrix is first generating a random m x m matrix A 
# and then taking the matrix A^T A (which is always semi-definite positive)
# m = 3
# A = np.random.rand(m, m)
# Sigma = A.T @ A

def markowitz_chanceconstraints(alpha, beta, mu, Sigma):

    model = pyo.ConcreteModel("Markowitz portfolio problem with chance constraints")

    model.x = pyo.Var(range(n), domain=pyo.NonNegativeReals)
    model.xtilde = pyo.Var(domain=pyo.NonNegativeReals)

    @model.Objective(sense=pyo.maximize)
    def objective(m):
        return mu @ m.x

    @model.Constraint()
    def chance_constraint(m):
        return norm.ppf(1-beta) * (m.x @ (Sigma @ m.x)) <= (mu @ m.x - alpha)

    @model.Constraint()
    def total_assets(m):
        return m.xtilde + sum(m.x[i] for i in range(n)) == C

    result = pyo.SolverFactory("cplex_direct").solve(model)

    return result, model

result, model = markowitz_chanceconstraints(alpha, beta, mu, Sigma)

display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:** $\\tilde x = {model.xtilde.value:.3f}$, $x_1 = {model.x[0].value:.3f}$,  $x_2 = {model.x[1].value:.3f}$,  $x_3 = {model.x[2].value:.3f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.objective():.2f}$"))

