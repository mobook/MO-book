#!/usr/bin/env python
# coding: utf-8

# # Two variants of the BIM problem: fractional objective and additional fixed costs

# In[10]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_glpk()


# Recall the BIM production model introduced earlier [here](BIM.ipynb), that is
# 
# $$
# \begin{array}{rrcrclr}
# \max \quad  
#         & 12x_1 & + & 9x_2               \\
# \text{s.t.} \quad
#         &   x_1 &   &      & \leq & 1000 &\text{(silicon)}\\
#         &       &   &  x_2 & \leq & 1500 &\text{(germanium)}\\
#         &   x_1 & + &  x_2 & \leq & 1750 &\text{(plastic)}\\
#         &  4x_1 & + & 2x_2 & \leq & 4800 &\text{(copper)}\\
#         &   x_1 & , &  x_2 & \geq & 0    \\
# \end{array}
# $$
# 
# Assume the pair $(12,9)$ reflects the sales price (revenues) in € and not the profits made per unit produced. We then need to account for the production costs. Suppose that the production costs for $(x_1,x_2)$ chips are equal to a fixed cost of 100 (independent of the number of units produced) plus $7/6 x_1$ plus $5/6 x_2$. It is reasonable to maximize the difference between the revenues and the costs. This approach yields the following linear model:

# In[11]:


import pyomo.environ as pyo

def BIM_with_revenues_minus_costs():
    
    m    = pyo.ConcreteModel('BIM with revenues minus costs')
    
    m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
    m.x2 = pyo.Var(domain=pyo.NonNegativeReals)

    m.revenue       = pyo.Expression( expr = 12*m.x1  +   9*m.x2 )
    m.variable_cost = pyo.Expression( expr = 7/6*m.x1 + 5/6*m.x2 )
    m.fixed_cost    = 100

    m.profit    = pyo.Objective( sense= pyo.maximize, expr = m.revenue - m.variable_cost - m.fixed_cost )

    m.silicon   = pyo.Constraint(expr =    m.x1          <= 1000)
    m.germanium = pyo.Constraint(expr =             m.x2 <= 1500)
    m.plastic   = pyo.Constraint(expr =    m.x1 +   m.x2 <= 1750)
    m.copper    = pyo.Constraint(expr =  4*m.x1 + 2*m.x2 <= 4800)

    return m

BIM_linear = BIM_with_revenues_minus_costs()
pyo.SolverFactory('glpk').solve(BIM_linear)

print('x=({:.1f},{:.1f}) value={:.3f} revenue={:.2f} cost={:.2f}'.format(
    pyo.value(BIM_linear.x1),
    pyo.value(BIM_linear.x2),
    pyo.value(BIM_linear.profit),
    pyo.value(BIM_linear.revenue),
    pyo.value(BIM_linear.variable_cost)+pyo.value(BIM_linear.fixed_cost)))


# This first model has the same optimal solution as the original BIM model, namely $(650,1100)$ with a revenue of $17700$ and a cost of $1775$.

# Alternatively, we may aim to optimize the efficiency of the plan, expressed as the ratio between the revenues and the costs:
# 
# $$
# \begin{array}{lll}
# \max \quad 
#         & {\dfrac{12x_1+9x_2}{7/6x_1 + 5/6x_2 + 100}} \\
# \text{s.t.} \quad 
#         &   x_1           \leq  1000 &\text{(silicon)}\\
#         &           x_2   \leq 1500  &\text{(germanium)}\\
#         &   x_1  +   x_2  \leq  1750 &\text{(plastic)}\\
#         &  4x_1  +  2x_2  \leq  4800 &\text{(copper)}\\
#         &   x_1  ,   x_2  \geq  0.
# \end{array}
# $$
# 
# In order to solve this second version we need to deal with the fraction appearing in the objective function by introducing an auxiliary variable $t \geq 0$. More specifically, we reformulate the model as follows
# 
# $$
# \begin{array}{rrcrcrclr}
# \max \quad 
#         & 12y_1 & + & 9y_2             \\
# \text{s.t.} \quad 
#         &   y_1 &   &       & & & \leq & 1000 \cdot t &\text{(silicon)}\\
#         &       &   &   y_2 & & & \leq & 1500 \cdot t &\text{(germanium)}\\
#         &   y_1 & + &   y_2 & & & \leq & 1750 \cdot t &\text{(plastic)}\\
#         &  4y_1 & + &  2y_2 & & & \leq & 4800 \cdot t &\text{(copper)}\\
# 		&7/6y_1 & + &5/6y_2 & + & 100y & = & 1 & \text{(fraction)} \\ 
#         &   y_1 & , &  y_2 & , & t & \geq & 0  \\
# \end{array}
# $$
# 
# Despite the change of variables, we can always recover the solution as $(x_1,x_2)= (y_1/t,y_2/t)$.

# In[12]:


def BIM_with_revenues_over_costs():
    
    m    = pyo.ConcreteModel('BIM with revenues over costs')
    
    m.y1 = pyo.Var(within=pyo.NonNegativeReals)
    m.y2 = pyo.Var(within=pyo.NonNegativeReals)
    m.t  = pyo.Var(within=pyo.NonNegativeReals)

    m.revenue       = pyo.Expression( expr = 12*m.y1  +   9*m.y2 )
    m.variable_cost = pyo.Expression( expr = 7/6*m.y1 + 5/6*m.y2 )
    m.fixed_cost    = 100

    m.profit    = pyo.Objective( sense= pyo.maximize, expr = m.revenue)

    m.silicon   = pyo.Constraint(expr =    m.y1          <= 1000*m.t)
    m.germanium = pyo.Constraint(expr =             m.y2 <= 1500*m.t)
    m.plastic   = pyo.Constraint(expr =    m.y1 +   m.y2 <= 1750*m.t)
    m.copper    = pyo.Constraint(expr =  4*m.y1 + 2*m.y2 <= 4800*m.t)
    m.frac      = pyo.Constraint(expr = m.variable_cost+m.fixed_cost*m.t == 1 )
    
    return m

BIM_fractional = BIM_with_revenues_over_costs()
pyo.SolverFactory('glpk').solve(BIM_fractional)

t = pyo.value(BIM_fractional.t)
print('x=({:.1f},{:.1f}) value={:.3f} revenue={:.2f} cost={:.2f}'.format(
    pyo.value(BIM_fractional.y1/t),
    pyo.value(BIM_fractional.y2/t),
    pyo.value(BIM_fractional.profit/(BIM_fractional.variable_cost+BIM_fractional.fixed_cost*t)),
    pyo.value(BIM_fractional.revenue/t),
    pyo.value(BIM_fractional.variable_cost/t)+pyo.value(BIM_fractional.fixed_cost)))


# The second model has optimal solution $(250,1500)$ with a revenue of $16500$ and a cost of $1641.667$.

# The efficiency, measured as the ratio revenue over costs for the optimal solution, is different for the two models. For the first model the efficiency is equal to $\frac{17700}{1775}=9.972$, which is strictly smaller than that of the second model, that is $\frac{16500}{1641.667}=10.051$.
