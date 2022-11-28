#!/usr/bin/env python
# coding: utf-8

# # BIM production

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## General LP formulation
# 
# The simplest and most scalable class of optimization problems is the one where the objective function and the constraints are formulated using the simplest possible type of functions - linear functions. A **linear program (LP)** is an optimization problem of the form
# 
# $$
# \begin{align*}
#     \min \quad & c^\top x\\
#     \text{s.t.} \quad & A x \leq b\\
#     & x \geq 0, \nonumber 
# \end{align*}
# $$
# 
# where the $n$ (decision) variables are grouped in a vector $x \in \mathbb{R}^n$, $c \in \mathbb{R}^n$ are the objective coefficients, and the $m$ linear constraints are described by the matrix $A \in \mathbb{R}^{m \times n}$ and the vector $b \in \mathbb{R}^m$. 
# 
# Of course, linear problems could also (i) be maximization problems, (ii) involve equality constraints and constraints of the form $\geq$, and (iii) have unbounded or non-positive decision variables $x_i$'s. In fact, any LP problem with such features can be easily converted to the 'canonical' LP form by adding/removing variables and/or multiplying specific inequalities by $-1$.

# ## The microchip production problem
# The company BIM (Best International Machines) produces two types of microchips, logic chips (1 g silicon, 1 g plastic, 4 g copper) and memory chips (1 g germanium, 1 g plastic, 2 g copper). Each of the logic chips can be sold for a 12 € profit, and each of the memory chips for a 9 € profit. The current stock of raw materials is as follows: 1000 g silicon, 1500 g germanium, 1750 g plastic, 4800 g of copper. How many microchips of each type should be produced to maximize the profit while respecting the raw material stock availability? 

# Let $x_1$ denote the number of logic chips and $x_2$ that of memory chips. This decision can be reformulated as an optimization problem of the following form:
# 
# $$
# \begin{align}
# \max  \quad  & 12 x_1 + 9 x_2 \\
# \text{s.t.} \quad
#     &   x_1 \leq 1000 &\text{silicon}\\
#     &   x_2 \leq 1500 &\text{germanium}\\
#     &   x_1 + x_2  \leq 1750 &\text{plastic}\\
#     &  4 x_1 + 2 x_2 \leq 4800 &\text{copper}\\
#     &   x_1, x_2 \geq 0 
# \end{align}
# $$

# The problem has $n=2$ decision variables and $m=4$ constraints. Using the standard notation introduced above, denote the vector of decision variables by $x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ and define the problem coefficients as
# 
# $$
# \begin{align*}
#     c = \begin{pmatrix} 12 \\ 9 \end{pmatrix},
#     \qquad
#     A = 
#     \begin{bmatrix}
#     1 & 0\\
#     0 & 1\\
#     1 & 1\\
#     4 & 2\\
#     \end{bmatrix},
#     \quad \text{ and } \quad
#     b = \begin{pmatrix} 1000 \\ 1500 \\ 1750 \\ 4800 \end{pmatrix}.
# \end{align*}
# $$
# 
# This model can be implemented and solved in Pyomo as follows.

# In[2]:


import pyomo.environ as pyo

m = pyo.ConcreteModel('BIM')

m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(domain=pyo.NonNegativeReals)

m.profit = pyo.Objective(expr=12*m.x1 + 9*m.x2, sense=pyo.maximize)

m.silicon = pyo.Constraint(expr=m.x1 <= 1000)
m.germanium = pyo.Constraint(expr=m.x2 <= 1500)
m.plastic = pyo.Constraint(expr=m.x1 + m.x2 <= 1750)
m.copper = pyo.Constraint(expr=4*m.x1 + 2*m.x2 <= 4800)

pyo.SolverFactory('cbc').solve(m)

print(f'x = ({m.x1.value:.1f}, {m.x2.value:.1f})')
print(f'optimal value = {pyo.value(m.profit):.2f}')


# ## Dual problem
# 
# One can construct bounds for the value of objective function by multiplying the constraints by non-negative numbers and adding them to each other so that the left-hand side looks like the objective function, while the right-hand side is the corresponding bound.
# 
# Let $\lambda_1,\lambda_2,\lambda_3,\lambda_4$ be non-negative numbers. If we multiply each of these variables by one of the four constraints of the original problem and sum all of them side by side to obtain the inequality
# 
# $$
# \begin{align*}
#         (\lambda_1+\lambda_3+4\lambda_4) x_1 + (\lambda_2+\lambda_3+2 \lambda_4) x_2 \leq 1000 \lambda_1 + 1500 \lambda_2 + 1750 \lambda_3 + 4800 \lambda_4.
# \end{align*}
# $$
# 
# It is clear that if $\lambda_1,\lambda_2,\lambda_3,\lambda_4 \geq 0$ satisfy
# 
# $$
# \begin{align*}
# \lambda_1+\lambda_3+4\lambda_4 & \geq 12,\\
# \lambda_2+\lambda_3+2 \lambda_4 & \geq 9,
# \end{align*}
# $$
# 
# then we have the following:
# 
# $$
# \begin{align*}
# 12 x_1 + 9 x_2 \leq (\lambda_1+\lambda_3+4\lambda_4) x_1 + (\lambda_2+\lambda_3+2 \lambda_4) x_2 \leq 1000 \lambda_1 + 1500 \lambda_2 + 1750 \lambda_3 + 4800 \lambda_4,
# \end{align*}
# $$
# 
# where the first inequality follows from the fact that $x_1, x_2 \geq 0$, and the most right-hand expression becomes an upper bound on the optimal value of the objective.
# 
# If we seek $\lambda_1,\lambda_2,\lambda_3,\lambda_4 \geq 0$ such that the upper bound on the RHS is as tight as possible, that means that we need to **minimize** the expression $1000 \lambda_1 + 1500 \lambda_2 + 1750 \lambda_3 + 4800 \lambda_4$. This can be formulated as the following LP, which we name the **dual problem**:
# 
# $$
# \begin{align*}
#         \min \quad & 1000 \lambda_1 + 1500 \lambda_2 + 1750 \lambda_3 + 4800 \lambda_4  \\
#         \text{s.t.} \quad & \lambda_1+\lambda_3+4\lambda_4 \geq 12,\\
#         & \lambda_2+\lambda_3+2 \lambda_4 \geq 9,\\
#         & \lambda_1,\lambda_2,\lambda_3,\lambda_4 \geq 0.
# \end{align*}
# $$
# 
# It is easy to solve and find the optimal solution $(\lambda_1,\lambda_2,\lambda_3,\lambda_4)=(0,0,6,1.5)$, for which the objective functions takes the value $17700$. Such a value is (the tightest) upper bound for the original problem. Here, we present the Pyomo code for this example.

# In[3]:


m = pyo.ConcreteModel('BIM dual')

m.y1 = pyo.Var(domain=pyo.NonNegativeReals)
m.y2 = pyo.Var(domain=pyo.NonNegativeReals)
m.y3 = pyo.Var(domain=pyo.NonNegativeReals)
m.y4 = pyo.Var(domain=pyo.NonNegativeReals)

m.obj = pyo.Objective(sense=pyo.minimize, 
                      expr=1000*m.y1 + 1500*m.y2 + 1750*m.y3 + 4800*m.y4)

m.x1 = pyo.Constraint(expr=m.y1 + m.y3 + 4*m.y4 >= 12)
m.x2 = pyo.Constraint(expr=m.y2 + m.y3 + 2*m.y4 >= 9)

pyo.SolverFactory('cbc').solve(m)
print(f'y = ({m.y1.value:.1f}, {m.y2.value:.1f}, {m.y3.value:.1f}, {m.y4.value:.1f})')
print(f"optimal value = {pyo.value(m.obj):.2f}")


# Note that since the original LP is feasible and bounded, strong duality holds and the optimal value of the primal problem coincides with the optimal value of the dual problem.

# In[ ]:




