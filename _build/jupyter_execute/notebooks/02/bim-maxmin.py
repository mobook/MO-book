#!/usr/bin/env python
# coding: utf-8

# ```{index} single: solver; cbc
# ```
# ```{index} single: solver; highs
# ```
# 
# # BIM production for worst case

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


# ## Minmax objective function
# 
# Another class of seemingly complicated objective functions that can be easily rewritten as an LP are those stated as maxima over several linear functions. Given a finite set of indices $K$ and a collection of vectors $\{c_k\}_{k \in K}$, the minimax problem given by
# 
# $$
# \begin{align}
#         \min \; \max_{k \in K} \; c^\top_{k} x
# \end{align}
# $$
# 
# General expressions like the latter can be linearized by introducing an auxiliary variable $z$ and setting
# 
# $$
# \begin{align*}
#     \min \quad & z  \\
#     \text{s.t.} \quad & c^\top_{k} x \leq z \qquad \forall\, k \in K.
# \end{align*}
# $$
# 
# This technique works because if *all* the quantities corresponding to different indices $k \in K$ are below the auxiliary variable $z$, then we are guaranteed that also their maximum is also below $z$ and vice versa. Note that the absolute value function can be rewritten $|x_i|= \max\{x_i, -x_i\}$, hence the linearization of the optimization problem involving absolute values in the objective functions is a special case of this. 
# 

# ## BIM problem variant: Maximizing the lowest possible profit
# 
# In the same way we can minimize a maximum like above, we can also maximize the minimum. Let us consider the [BIM microchip production problem](bim.ipynb), but suppose that there is uncertainty regarding the selling prices of the microchips. Instead of just the nominal prices 12 € and 9 €, BIM estimates that prices may more generally take values $P=\{ (12,9), (11,10), (8, 11) \}$. The optimization problem for a production plan that achieves the maximum among the lowest possible profits can be formulated using the trick mentioned above and can be implemented in Pyomo as follows.

# In[2]:


import pyomo.environ as pyo

def BIM_maxmin(costs):
    
    m = pyo.ConcreteModel('BIM with maxmin objective')
    
    m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
    m.x2 = pyo.Var(domain=pyo.NonNegativeReals)
    m.z = pyo.Var() 

    m.profit = pyo.Objective(sense= pyo.maximize, expr = m.z)

    m.maxmin = pyo.ConstraintList()
    for (c1, c2) in costs:
        m.maxmin.add(expr = m.z <= c1*m.x1 + c2*m.x2) 

    m.silicon = pyo.Constraint(expr = m.x1 <= 1000)
    m.germanium = pyo.Constraint(expr = m.x2 <= 1500)
    m.plastic = pyo.Constraint(expr = m.x1 + m.x2 <= 1750)
    m.copper = pyo.Constraint(expr = 4*m.x1 + 2*m.x2 <= 4800)

    return m
    
BIM = BIM_maxmin([[12, 9], [11, 10], [8, 11]])

SOLVER.solve(BIM)

print(f"x = ({pyo.value(BIM.x1):.1f}, {pyo.value(BIM.x2):.1f})")
print(f"revenue = {pyo.value(BIM.profit):.2f}")

