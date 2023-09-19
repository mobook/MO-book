#!/usr/bin/env python
# coding: utf-8

# ```{index} dual problem
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} single: application; production planning
# ```
# 
# # Dual of the BIM production problem

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


# ## Derivation of the dual problem
# 
# In a [previous notebook](bim.ipynb), we introduce the BIM production problem and showed that it can be modeled as the following LO problem:
# $$
# \begin{align*}
# \max  \quad  & 12 x_1 + 9 x_2 \\
# \text{s.t.} \quad
#     &   x_1 \leq 1000 &\text{(silicon)}\\
#     &   x_2 \leq 1500 &\text{(germanium)}\\
#     &   x_1 + x_2  \leq 1750 &\text{(plastic)}\\
#     &  4 x_1 + 2 x_2 \leq 4800 &\text{(copper)}\\
#     &   x_1, x_2 \geq 0.
# \end{align*}
# $$
# In this notebook, we will derive step by step its **dual problem**. 
# 
# One can construct bounds for the value of objective function of the original problem by multiplying the constraints by non-negative numbers and adding them to each other so that the left-hand side looks like the objective function, while the right-hand side is the corresponding bound.
# 
# Let $\lambda_1,\lambda_2,\lambda_3,\lambda_4$ be non-negative numbers. If we multiply each of these variables by one of the four constraints of the original problem and sum all of them side by side to obtain the inequality
# 
# $$
# \begin{align*}
# (\lambda_1+\lambda_3+4\lambda_4) x_1 + (\lambda_2+\lambda_3+2 \lambda_4) x_2 \leq 1000 \lambda_1 + 1500 \lambda_2 + 1750 \lambda_3 + 4800 \lambda_4.
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
# If we seek $\lambda_1,\lambda_2,\lambda_3,\lambda_4 \geq 0$ such that the upper bound on the RHS is as tight as possible, that means that we need to **minimize** the expression $1000 \lambda_1 + 1500 \lambda_2 + 1750 \lambda_3 + 4800 \lambda_4$. This can be formulated as the following LO, which we name the **dual problem**:
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
# It is easy to solve and find the optimal solution $(\lambda_1,\lambda_2,\lambda_3,\lambda_4)=(0,0,6,1.5)$, for which the objective functions takes the value $17700$. Such a value is (the tightest) upper bound for the original problem. 
# 
# The Pyomo code that implements and solves the dual problem is given below. 

# In[3]:


import pyomo.environ as pyo

model = pyo.ConcreteModel('BIM dual')

# Decision variables and their domain
model.y1 = pyo.Var(domain=pyo.NonNegativeReals)
model.y2 = pyo.Var(domain=pyo.NonNegativeReals)
model.y3 = pyo.Var(domain=pyo.NonNegativeReals)
model.y4 = pyo.Var(domain=pyo.NonNegativeReals)

# Objective function
model.obj = pyo.Objective(sense=pyo.minimize, 
                      expr=1000*model.y1 + 1500*model.y2 + 1750*model.y3 + 4800*model.y4)

# Constraints
model.x1 = pyo.Constraint(expr=model.y1 + model.y3 + 4*model.y4 >= 12)
model.x2 = pyo.Constraint(expr=model.y2 + model.y3 + 2*model.y4 >= 9)

# Solve and print solution
SOLVER.solve(model)
print(f'y = ({model.y1.value:.1f}, {model.y2.value:.1f}, {model.y3.value:.1f}, {model.y4.value:.1f})')
print(f"optimal value = {pyo.value(model.obj):.1f}")


# Note that since the original LO is feasible and bounded, strong duality holds and the optimal value of the primal problem coincides with the optimal value of the dual problem.
