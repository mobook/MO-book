#!/usr/bin/env python
# coding: utf-8

# ```{index} single: solver; cbc
# ```
# ```{index} single: application; production planning
# ```
# 
# # BIM production with perturbed data

# ## Preamble: Install Pyomo and a solver
# 
# This cell selects and verifies a global SOLVER for the notebook.
# 
# If run on Google Colab, the cell installs Pyomo and HiGHS, then sets SOLVER to 
# use the Highs solver via the appsi module. If run elsewhere, it assumes Pyomo and CBC
# have been previously installed and sets SOLVER to use the CBC solver via the Pyomo 
# SolverFactory. It then verifies that SOLVER is available.

# In[1]:


import sys

if 'google.colab' in sys.modules:
    get_ipython().system('pip install pyomo >/dev/null 2>/dev/null')
    get_ipython().system('pip install highspy >/dev/null 2>/dev/null')

    from pyomo.contrib import appsi
    SOLVER = appsi.solvers.Highs(only_child_vars=False)
    
else:
    from pyomo.environ import SolverFactory
    SOLVER = SolverFactory('cbc')

assert SOLVER.available(), f"Solver {SOLVER} is not available."


# ## Problem description
# 
# The company BIM realizes that a $1\%$ fraction of the copper always gets wasted while producing both types of microchips, more specifically $1\%$ of the required amount. This means that it actually takes $4.04$ gr of copper to produce a logic chip and $2.02$ gr of copper to produce a memory chip. If we rewrite the linear problem of [the basic BIM problem](../02/bim.ipynb) and modify accordingly the coefficients in the corresponding constraints, we obtain the following problem 
# 
# $$
# \begin{align*}
#         \max \quad & 12x_1+9x_2  \\
#         \text{s.t.} \quad & x_1 \leq 1000 & \text{(silicon)}\\
#         & x_2 \leq 1500 & \text{(germanium)}\\
#         & x_1+x_2 \leq 1750 & \text{(plastic)}\\
#         & 4.04 x_1+2.02 x_2 \leq 4800 & \text{(copper with waste)}\\
#         &x_1, x_2 \geq 0.
# \end{align*}
# $$
# 
# If we solve it, we obtain a different optimal solution than the original one, namely $(x_1,x_2) \approx (626.238,1123.762)$ and an optimal value of roughly $17628.713$. The new optimal solution is not integral but, in fact, there is no constraint requiring $x_1$ and $x_2$ to be integers.

# In[2]:


import pyomo.environ as pyo

m    = pyo.ConcreteModel('BIM perturbed LO')

m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(domain=pyo.NonNegativeReals)

m.profit    = pyo.Objective( expr = 12*m.x1 + 9*m.x2, sense= pyo.maximize)

m.silicon   = pyo.Constraint(expr =       m.x1             <= 1000)
m.germanium = pyo.Constraint(expr =                   m.x2 <= 1500)
m.plastic   = pyo.Constraint(expr =       m.x1 +      m.x2 <= 1750)
m.copper    = pyo.Constraint(expr =  4.04*m.x1 + 2.02*m.x2 <= 4800)

SOLVER.solve(m)

print('x = ({:.3f}, {:.3f}), optimal value = {:.3f}'.format(
    pyo.value(m.x1),
    pyo.value(m.x2),
    pyo.value(m.profit)))


# In terms of production, we want to manufacture an integer number of microchips, but it is not clear how to implement the fractional optimal solution $(x_1,x_2) \approx (626.238,1123.762)$. Rounding down to $(x_1,x_2) = (626,1123)$ will intuitively yield a feasible solution, but this could lead to a minor loss of profit and/or an inefficient use of the available material. Rounding up to $(x_1,x_2) = (627,1124)$ could possibly lead to an unfeasible solution for which the available material is not enough. We could, of course, examine all the potential integer solutions by hand. However, if the problem had a more intricate structure or a greater number of decision variables, this would be much more difficult and may not lead to the true optimal solution.
# 
# A safer approach is to explicitly require the two decision variables to be non-negative integers, thus transforming the original LO problem into the following mixed-integer linear optimization (MILO) problem:
# 
# $$
# \begin{align*}
#         \max \quad & 12x_1+9x_2  \\
#         \text{s.t.} \quad & x_1 \leq 1000 & \text{(silicon)}\\
#         & x_2 \leq 1500 & \text{(germanium)}\\
#         & x_1+x_2 \leq 1750 & \text{(plastic)}\\
#         & 4.04 x_1+2.02 x_2 \leq 4800 & \text{(copper with waste)}\\
#         &x_1, x_2 \in \mathbb{N}.
# \end{align*}
# $$
# 
# The optimal solution of this new MILO problem is $(x_1,x_2) = (626,1124)$ with a profit of $17628$. Note that for this specific problem both the naive rounding strategies outlined above would not have yielded the true optimal solution. The Python code for obtaining the optimal solution using MILO solvers is given below.

# In[3]:


m    = pyo.ConcreteModel('BIM perturbed MILO')

m.x1 = pyo.Var(domain=pyo.NonNegativeIntegers)
m.x2 = pyo.Var(domain=pyo.NonNegativeIntegers)

m.profit    = pyo.Objective( expr = 12*m.x1 + 9*m.x2, sense= pyo.maximize)

m.silicon   = pyo.Constraint(expr =       m.x1             <= 1000)
m.germanium = pyo.Constraint(expr =                   m.x2 <= 1500)
m.plastic   = pyo.Constraint(expr =       m.x1 +      m.x2 <= 1750)
m.copper    = pyo.Constraint(expr =  4.04*m.x1 + 2.02*m.x2 <= 4800)

SOLVER.solve(m)

print('x = ({:.3f}, {:.3f}), optimal value = {:.3f}'.format(
    pyo.value(m.x1),
    pyo.value(m.x2),
    pyo.value(m.profit)))

