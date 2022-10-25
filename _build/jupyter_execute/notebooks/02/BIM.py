#!/usr/bin/env python
# coding: utf-8

# # A first LP example: the microchip production problem of the BIM company

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
# The company BIM (Best International Machines) produces two types of microchips, logic chips (1gr silicon, 1gr plastic, 4gr copper) and memory chips (1gr germanium, 1gr plastic, 2gr copper). Each of the logic chips can be sold for a 12 € profit, and each of the memory chips for a 9 € profit. The current stock of raw materials is as follows: 1000gr silicon, 1500gr germanium, 1750gr plastic, 4800gr of copper. How many microchips of each type should be produced to maximize the profit while respecting the raw material stock availability? 

# Let $x_1$ denote the number of logic chips and $x_2$ that of memory chips. This decision can be reformulated as an optimization problem of the following form:
# 
# $$
# \begin{array}{rrcrclr}
# \max  \quad  & 12x_1 & + & 9x_2               \\
# \text{s.t.}
#     &   x_1 &   &      & \leq & 1000 &\text{(silicon)}\\
#     &       &   &  x_2 & \leq & 1500 &\text{(germanium)}\\
#     &   x_1 & + &  x_2 & \leq & 1750 &\text{(plastic)}\\
#     &  4x_1 & + & 2x_2 & \leq & 4800 &\text{(copper)}\\
#     &   x_1 & , &  x_2 & \geq & 0.    
# \end{array}
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

# In[27]:


import pyomo.environ as pyo

m    = pyo.ConcreteModel('BIM')

m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(domain=pyo.NonNegativeReals)

m.profit    = pyo.Objective( expr = 12*m.x1 + 9*m.x2, sense= pyo.maximize)

m.silicon   = pyo.Constraint(expr =    m.x1          <= 1000)
m.germanium = pyo.Constraint(expr =             m.x2 <= 1500)
m.plastic   = pyo.Constraint(expr =    m.x1 +   m.x2 <= 1750)
m.copper    = pyo.Constraint(expr =  4*m.x1 + 2*m.x2 <= 4800)

pyo.SolverFactory('glpk').solve(m)

print('x = ({:.1f},{:.1f}) optimal value = {:.2f}'.format(
    pyo.value(m.x1),
    pyo.value(m.x2),
    pyo.value(m.profit)))

