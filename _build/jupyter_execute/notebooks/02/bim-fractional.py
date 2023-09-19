#!/usr/bin/env python
# coding: utf-8

# ```{index} single: solver; cbc
# ```
# ```{index} single: solver; highs
# ```
# 
# # BIM production variants

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


# ## Two variants of the BIM problem: fractional objective and additional fixed costs
# 
# Recall the BIM production model introduced earlier [here](bim.ipynb), that is
# 
# $$
# \begin{align*}
#     \max \quad & 12x_1 + 9x_2 \\
#     \textup{s.t.} \quad & x_1 \leq 1000 \quad &\textup{(silicon)} \\
#                        & x_2 \leq 1500 \quad &\textup{(germanium)} \\
#                        & x_1 + x_2 \leq 1750 \quad &\textup{(plastic)} \\
#                        & 4x_1 + 2x_2 \leq 4800 \quad &\textup{(copper)} \\
#                        & x_1, x_2 \geq 0. \\
# \end{align*}
# $$
# 
# Assume the pair $(12,9)$ reflects the sales price (revenues) in € and not the profits made per unit produced. We then need to account for the production costs. Suppose that the production costs of the chips $(x_1,x_2)$ are equal to a fixed cost of $100$ (independent of the number of units produced) plus $7/6x_1$ plus $5/6x_2$. It is reasonable to maximize the difference between revenues and costs. This approach yields the following linear model:
# 
# $$
# \begin{align*}
#     \max \quad & \left (12-\frac{7}{6} \right)x_1 + \left (9-\frac{5}{6} \right)x_2 - 100 \\
#     \textup{s.t.} \quad 
#         & x_1 \leq 1000 \quad &\textup{(silicon)} \\
#         & x_2 \leq 1500 \quad &\textup{(germanium)} \\
#         & x_1 + x_2 \leq 1750 \quad &\textup{(plastic)} \\
#         & 4x_1 + 2x_2 \leq 4800 \quad &\textup{(copper)} \\
#         & x_1, x_2 \geq 0.
# \end{align*}
# $$

# In[2]:


import pyomo.environ as pyo

def BIM_with_revenues_minus_costs():
    
    m = pyo.ConcreteModel('BIM with revenues minus costs')
    
    m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
    m.x2 = pyo.Var(domain=pyo.NonNegativeReals)

    m.revenue = pyo.Expression(expr = 12 * m.x1 + 9 * m.x2)
    m.variable_cost = pyo.Expression(expr = 7/6 * m.x1 + 5/6 * m.x2)
    m.fixed_cost = 100

    m.profit = pyo.Objective(sense= pyo.maximize, 
                             expr = m.revenue - m.variable_cost - m.fixed_cost)

    m.silicon = pyo.Constraint(expr = m.x1 <= 1000)
    m.germanium = pyo.Constraint(expr = m.x2 <= 1500)
    m.plastic = pyo.Constraint(expr = m.x1 + m.x2 <= 1750)
    m.copper = pyo.Constraint(expr = 4*m.x1 + 2*m.x2 <= 4800)

    return m

BIM_linear = BIM_with_revenues_minus_costs()
SOLVER.solve(BIM_linear)

print('x=({:.1f},{:.1f}) value={:.3f} revenue={:.2f} cost={:.2f}'.format(
    pyo.value(BIM_linear.x1),
    pyo.value(BIM_linear.x2),
    pyo.value(BIM_linear.profit),
    pyo.value(BIM_linear.revenue),
    pyo.value(BIM_linear.variable_cost) + pyo.value(BIM_linear.fixed_cost)))


# This first model has the same optimal solution as the original BIM model, namely $(650,1100)$ with a revenue of $17700$ and a cost of $1775$.

# Alternatively, we may aim to optimize the efficiency of the plan, expressed as the ratio between the revenues and the costs:
# 
# $$
# \begin{align*}
#     \max \quad & \dfrac{12x_1 + 9x_2}{\frac{7}{6}x_1 + \frac{5}{6}x_2 + 100} \\
#     \textup{s.t.} \quad 
#         & x_1 \leq 1000 \quad &\textup{(silicon)} \\
#         & x_2 \leq 1500 \quad &\textup{(germanium)} \\
#         & x_1 + x_2 \leq 1750 \quad &\textup{(plastic)} \\
#         & 4x_1 + 2x_2 \leq 4800 \quad &\textup{(copper)} \\
#         & x_1, x_2 \geq 0.
# \end{align*}
# $$
# 
# In order to solve this second version we need to deal with the fraction appearing in the objective function by introducing an auxiliary variable $t \geq 0$. More specifically, we reformulate the model as follows
# 
# $$
# \begin{align*}
#     \max \quad & 12y_1 + 9y_2 \\
#     \textup{s.t.} \quad 
#         & y_1 \leq 1000 \cdot t \quad &\textup{(silicon)} \\
#         & y_2 \leq 1500 \cdot t \quad &\textup{(germanium)} \\
#         & y_1 + y_2 \leq 1750 \cdot t \quad &\textup{(plastic)} \\
#         & 4y_1 + 2y_2 \leq 4800 \cdot t \quad &\textup{(copper)} \\
#         & \frac{7}{6}y_1 + \frac{5}{6}y_2 + 100y = 1 \quad &\textup{(fraction)} \\
#         & y_1, y_2, t \geq 0.
# \end{align*}
# $$
# 
# Despite the change of variables, we can always recover the solution as $(x_1,x_2)= (y_1/t,y_2/t)$.

# In[3]:


def BIM_with_revenues_over_costs():
    
    m = pyo.ConcreteModel('BIM with revenues over costs')
    
    m.y1 = pyo.Var(domain=pyo.NonNegativeReals)
    m.y2 = pyo.Var(domain=pyo.NonNegativeReals)
    m.t = pyo.Var(domain=pyo.NonNegativeReals)

    m.revenue = pyo.Expression(expr = 12 * m.y1 + 9 * m.y2 )
    m.variable_cost = pyo.Expression(expr = 7/6 * m.y1 + 5/6 * m.y2)
    m.fixed_cost = 100

    m.profit = pyo.Objective(sense= pyo.maximize, expr = m.revenue)

    m.silicon = pyo.Constraint(expr = m.y1 <= 1000 * m.t)
    m.germanium = pyo.Constraint(expr = m.y2 <= 1500 * m.t)
    m.plastic = pyo.Constraint(expr = m.y1 + m.y2 <= 1750 * m.t)
    m.copper = pyo.Constraint(expr = 4*m.y1 + 2*m.y2 <= 4800 * m.t)
    m.frac = pyo.Constraint(expr = m.variable_cost + m.fixed_cost * m.t == 1)
    
    return m

BIM_fractional = BIM_with_revenues_over_costs()
SOLVER.solve(BIM_fractional)

t = pyo.value(BIM_fractional.t)
print('x=({:.1f},{:.1f}) value={:.3f} revenue={:.2f} cost={:.2f}'.format(
    pyo.value(BIM_fractional.y1 / t),
    pyo.value(BIM_fractional.y2 / t),
    pyo.value(BIM_fractional.profit / (BIM_fractional.variable_cost + BIM_fractional.fixed_cost * t)),
    pyo.value(BIM_fractional.revenue / t),
    pyo.value(BIM_fractional.variable_cost / t) + pyo.value(BIM_fractional.fixed_cost)))


# The second model has optimal solution $(250,1500)$ with a revenue of $16500$ and a cost of $1641.667$.
# 
# The efficiency, measured as the ratio of revenue over costs for the optimal solution, is different for the two models. For the first model the efficiency is equal to $\frac{17700}{1775}=9.972$, which is strictly smaller than that of the second model, that is $\frac{16500}{1641.667}=10.051$.
