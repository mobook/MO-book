#!/usr/bin/env python
# coding: utf-8

# ```{index} single: Pyomo; block
# ```
# ```{index} single: Pyomo; kernel library
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} single: solver; ipopt
# ```
# ```{index} single: solver; cplex
# ```
# ```{index} single: solver; gurobi
# ```
# ```{index} single: solver; xpress
# ```
# ```{index} data-driven uncertainty sets
# ```
# ```{index} single: application; production planning
# ```
# ```{index} robust optimization
# ```
# 
# # Robust BIM microchip production problem
# 

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()
helper.install_ipopt()
helper.install_gurobi()
helper.install_xpress()
helper.install_cplex()

import pyomo.environ as pyo
cbc    = pyo.SolverFactory('cbc')
ipopt  = pyo.SolverFactory('ipopt')
cplex  = pyo.SolverFactory('cplex_direct')
gurobi = pyo.SolverFactory('gurobi_direct')
xpress = pyo.SolverFactory('xpress_direct')


# ## Original BIM production planning model
# 
# The full description of the BIM production problem, can be found here [here](../02/bim.ipynb). The resulting optimization problem was the following LP:
# 
# $$
# \begin{array}{rrcrcl}
# \max    & 12x_1 & + & 9x_2               \\
# \text{s.t.}    &   x_1 &   &      & \leq & 1000 \\
#         &       &   &  x_2 & \leq & 1500 \\
#         &   x_1 & + &  x_2 & \leq & 1750 \\
#         &  4x_1 & + & 2x_2 & \leq & 4800 \\
#         &   x_1 & , &  x_2 & \geq & 0.
# \end{array}
# $$

# In[2]:


chips = [ 'logic', 'memory' ]
profits  = { 'logic' : 12, 'memory' :  9 }
copper     = { 'logic' :  4, 'memory' :  2 }

m = pyo.ConcreteModel('BIM basic problem')

m.chips = pyo.Set(initialize=chips)
m.x = pyo.Var(m.chips,within=pyo.NonNegativeReals)

m.profit    = pyo.Objective(expr = pyo.quicksum([profits[c]*m.x[c] for c in m.chips]), sense=pyo.maximize)

m.silicon  = pyo.Constraint(expr = m.x['logic']  <= 1000)
m.gemanium = pyo.Constraint(expr = m.x['memory'] <= 1500)
m.plastic  = pyo.Constraint(expr = pyo.quicksum([m.x[c] for c in m.chips]) <= 1750)
m.copper   = pyo.Constraint(expr = pyo.quicksum(copper[c]*m.x[c] for c in m.chips) <= 4800) 

cbc.solve(m)
print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# In[3]:


def ShowDuals(model):
    import fractions
    # display all duals
    print("The dual variable corresponding to:\n")
    for c in model.component_objects(pyo.Constraint, active=True):
        print("- the constraint on",c, "is equal to ", end="")
        for index in c:
            print(str(fractions.Fraction(model.dual[c[index]])))
            
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

cbc.solve(m)
ShowDuals(m)


# ## Robust BIM production planning models
# 
# Suppose now that there is uncertainty affecting the microchip production at BIM. Specifically, the company notices that not the amount of copper needed for the two types of microchips is not _exactly_ 4 and 2 gr, but varies due to some external factors affecting the production process. How does this uncertainty affect the optimal production plan?

# To get a feeling for what happens, let us first perform some simulations and data analysis on them. We start by simulating a sample of $n=2000$ observed copper consumption pairs for the production of `f` logic chips and `g` memory chips. The amounts vary around the original values, 4 gr and 2 gr, respectively, according to two independent lognormal distributions.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams.update({'font.size': 12})

seed = 0
rng = np.random.default_rng(seed)
n = 2000

f = rng.lognormal(np.log(4.), .005, n)
g = rng.lognormal(np.log(2.), .005, n)

plt.figure()
plt.plot(f, g, '.')
plt.xlabel("Copper gr needed for logic chips")
plt.ylabel("Copper gr needed for memory chips")
plt.show()


# ### Box uncertainty for copper consumption
# 
# A very simple and somehow naive uncertainty set can be the minimal box that contains all the simulated data.

# In[5]:


plt.figure()
plt.plot(f, g, '.')
currentAxis = plt.gca()
currentAxis.add_patch(patches.Rectangle((min(f), min(g)), max(f)-min(f), max(g)-min(g),fill=False,color='r'))
plt.xlabel("Copper gr needed for logic chips")
plt.ylabel("Copper gr needed for memory chips")
plt.show()

# calculate the upper and lower bounds for each uncertain parameter
lower = {'logic': min(f), 'memory': min(g)}
upper = {'logic': max(f), 'memory': max(g)}
print('Lower bounds',lower)
print('Upper bounds',upper)


# Using this empirical box uncertainty set, we can consider the following robust variant of their optimization model:
# 
# $$
# \begin{array}{rrcrcl}
# \max    &  12 x_1 & + &   9 x_2               \\
# \text{s.t.}    &     x_1 &   &         & \leq & 1000 \\
#         &         &   &     x_2 & \leq & 1500 \\
#         &     x_1 & + &     x_2 & \leq & 1750 \\
#         & z_1 x_1 & + & z_2 x_2 & \leq & 4800 & \forall \ell \leq a \leq u \\
#         &     x_1 & , &     x_2 & \geq & 0    \\
# \end{array}
# $$
# 
# The above model has an infinite number of constraints, one for every realization of the uncertain coefficients $z$. However, using linear duality, we can deal with this and obtain a robustified LP that we can solve. 

# ### Robust counterpart of box uncertainty
# 
# The first thing to notice is that the copper consumption is modeled by constraints that are equivalent to bounding the following optimization problem:
#     
# $$
# \begin{array}{rrr}
# \max    & x_1 z_1 + x_2 z_2 & \leq 4800 \\
# \text{s.t.}    & \ell \leq z \leq u 
# \end{array}
# $$
# 
# or
# 
# $$
# \begin{array}{rrr}
# \max    & x_1 z_1 + x_2 z_2 & \leq 4800 \\
# \text{s.t.}    & z \leq u \\
#         & -z \leq -\ell.
# \end{array}
# $$
# 
# Now we use linear duality to realize that the above is equivalent to:
#  
# $$
# \begin{array}{rrr}
# \min    & u y  - \ell w & \leq 4800 \\
# \text{s.t.}    & y - w = x \\
#         & y \geq 0, w \geq 0
# \end{array}
# $$
#     
# and the constraint imposed by the last problem is equivalent to:
# 
# $$
# \begin{array}{rrl}
#    & u y  - \ell w & \leq 4800 \\
#    & y - w & = x \\
#    & y \geq 0, w \geq 0
# \end{array}
# $$
# 
# The only thing we need to do is add the new auxiliary variables and constraints to the original model and implement them in Pyomo.

# In[6]:


def BIMWithBoxUncertainty(lower, upper, domain=pyo.NonNegativeReals):

    m = pyo.ConcreteModel('BIM with Box Uncertainty')
    
    m.chips = pyo.Set(initialize=chips)
    m.x = pyo.Var(m.chips,within=domain)

    m.profit    = pyo.Objective(expr = sum([profits[c]*m.x[c] for c in m.chips]), sense=pyo.maximize)

    m.silicon   = pyo.Constraint(expr = m.x['logic']  <= 1000)
    m.germanium = pyo.Constraint(expr = m.x['memory'] <= 1500)
    m.plastic   = pyo.Constraint(expr = sum([m.x[c] for c in m.chips]) <= 1750)

    m.y = pyo.Var(m.chips,domain=pyo.NonNegativeReals)
    m.w = pyo.Var(m.chips,domain=pyo.NonNegativeReals)

    m.robustcopper = pyo.Constraint(expr = sum([upper[c]*m.y[c] - lower[c]*m.w[c] for c in m.chips]) <= 4800)
    
    @m.Constraint(m.chips)
    def PerVariable(m, c):
        return m.x[c] == m.y[c] - m.w[c]
    
    return m

m = BIMWithBoxUncertainty(lower, upper)
cbc.solve(m)

print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# We may want to impose the box uncertainty set to be symmetric with respect to the nominal values and just choose its width $\delta$. This leads to a different optimal robust solution.

# In[7]:


# The parameter delta allows you to tune the amount of uncertainty. 
# In particular, if you take delta=0, you obtain the same result as the nominal model.
delta = 0.05

def BIMWithSymmetricalBoxUncertainty(delta, domain=pyo.NonNegativeReals):
    lower = { chip : copper[chip] - delta for chip in chips }
    upper = { chip : copper[chip] + delta for chip in chips }
    return BIMWithBoxUncertainty(lower, upper, domain=domain)

m = BIMWithSymmetricalBoxUncertainty(delta)
cbc.solve(m)

print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# ### Integer solution variant
# 
# The original BIM model gave integer solutions, but not the robust version. If we need integer solutions then we should impose that to the nature of the variables, which in this case of _box uncertainty_ is easy to do since the model remains linear, although it will be mixed integer. 

# In[8]:


m = BIMWithBoxUncertainty(lower, upper, domain=pyo.NonNegativeIntegers)
cbc.solve(m)

print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# Let us see how the optimal solution behave as we vary the width of the box uncertainty set $\delta$ from 0 to 0.5. 

# In[9]:


import pandas as pd

df = pd.DataFrame()
for delta in np.linspace(0, 0.5, 21):
    m = BIMWithSymmetricalBoxUncertainty(delta, domain=pyo.NonNegativeIntegers)
    cbc.solve(m)
    results = [ pyo.value(m.profit) ] + [ pyo.value(m.x[i]) for i in m.chips ]
    df.at[delta,'profit']     = results[0]
    df.at[delta,chips[0]] = results[1]
    df.at[delta,chips[1]] = results[2]
df


# We can visualize how these quantities change as a function of $\delta$:

# In[10]:


df[['profit']].plot()
plt.ylim([16001, 17999])
plt.xlabel("Margin $\delta$ of the uncertainty box")
plt.ylabel("Profit")
plt.show()
df[['logic','memory']].plot()
plt.xlabel("Margin $\delta$ of the uncertainty box")
plt.ylabel("Optimal number of produced chips")
plt.show()


# ## Cardinality-constrained uncertainty set
# 
# Let us now make different assumptions regarding the uncertainty related to the copper consumption. More specifically, we now assume that each uncertain coefficient $z_j$ may deviate by at most $\pm \delta$ from the nominal value $\bar{z}_j$ but no more than $\Gamma$ will actually deviate.
# 
# $$
# \begin{array}{rrcrcl}
# \max    &  12 x_1 & + &   9 x_2               \\
# \text{s.t.}    &     x_1 &   &         & \leq & 1000 \\
#         &         &   &     x_2 & \leq & 1500 \\
#         &     x_1 & + &     x_2 & \leq & 1750 \\
#         & z_1 x_1 & + & z_2 x_2 & \leq & 4800 & \forall \, y \in \mathbb{R}^2 \,:\, z_j=\bar{z}_j+\delta y_j, \, \|y\|_\infty \leq 1, \, \|y\|_1\leq \Gamma \\
#         &     x_1 & , &     x_2 & \geq & 0    \\
# \end{array}
# $$

# ### Robust counterpart of cardinality-constrained uncertainty
# Lagrange duality yields the following modification to the problem as equivalent to the robust model stated above:
# 
# $$
# \begin{array}{rrcrcrcrcrcrcl}
# \max    &  12 x_1 & + &   9 x_2               \\
# \text{s.t.}    &     x_1 &   &         & & & & & & & \leq & 1000 \\
#         &         &   &     x_2 & & & & & & & \leq & 1500 \\
#         &     x_1 & + &     x_2 & & & & & & & \leq & 1750 \\
#         & \bar{z}_1 x_1 & + & \bar{z}_2 x_2 & + & \lambda\Gamma & + & t_1 & + & t_2 & \leq & 4800 \\
#         &-\delta x_1 &   &         & + & \lambda & + & t_1 &   &     & \geq & 0 \\
#         &         &   &-\delta x_2 & + & \lambda &   &     & + & t_2 & \geq & 0 \\
#         &\delta x_1 &   &         & + & \lambda & + & t_1 &   &     & \geq & 0 \\
#         &         &   &\delta x_2 & + & \lambda &   &     & + & t_2 & \geq & 0 \\
#         &     x_1 & , &     x_2 & , & \lambda & , & t_1 & , & t_2 & \geq & 0    \\
# \end{array}
# $$

# In[11]:


def BIMWithBudgetUncertainty(delta, gamma, domain=pyo.NonNegativeReals):
    
    m = pyo.ConcreteModel('BIM with Budget Uncertainty')
    
    m.chips = pyo.Set(initialize=chips)
    m.x = pyo.Var(m.chips, domain=domain)

    m.profit    = pyo.Objective(expr = sum([profits[c]*m.x[c] for c in m.chips]), sense=pyo.maximize)

    m.silicon   = pyo.Constraint(expr = m.x['logic']  <= 1000)
    m.germanium = pyo.Constraint(expr = m.x['memory'] <= 1500)
    m.plastic   = pyo.Constraint(expr = sum([m.x[c] for c in m.chips]) <= 1750)

    m.t     = pyo.Var(m.chips, domain=pyo.NonNegativeReals)
    m.lam   = pyo.Var(domain=pyo.NonNegativeReals)

    m.robustcopper = pyo.Constraint(expr = sum([copper[c]*m.x[c] for c in m.chips]) + gamma * m.lam + sum(m.t[c] for c in m.chips) <= 4800)

    @m.Constraint(m.chips)
    def up_rule(m, c):
        return m.t[c] >= delta * m.x[c] - m.lam
    
    @m.Constraint(m.chips)
    def down_rule(m, c):
        return m.t[c] >= -delta * m.x[c] - m.lam

    return m


# In[12]:


m = BIMWithBudgetUncertainty(0.01, 2, domain=pyo.NonNegativeIntegers)
cbc.solve(m)

print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# ## Ball uncertainty set
# 
# Let us now make yet another different assumption regarding the uncertainty related to copper consumption. More specifically, we assume that the two uncertain coefficients $z_1$ and $z_2$ can vary in a 2-dimensional ball centered around the point $(\bar{z}_1,\bar{z}_2) = (4,2)$ and with radius $r$. 
# 
# ### Robust counterpart of ball uncertainty
# A straightforward reformulation leads to the equivalent constraint:
# 
# $$
#   \bar{z}_1x_1+\bar{z}_2x_2 + r \|x\|_2 \leq 4800
# $$
# 
# By defining $y = 4800 - \bar{z}_1x_1 - \bar{z}_2x_2$ and $w = r x$, we may write:
# 
# $$
#   \|w\|^2_2 \leq y^2
# $$

# We now need to add this newly obtained conic constraint to the original BIM model. The [Pyomo documentation](https://pyomo.readthedocs.io/en/stable/library_reference/kernel/conic.html) says a conic constraint is expressed in 'pyomo' in simple variables and this [table](https://pyomo.readthedocs.io/en/stable/library_reference/kernel/syntax_comparison.html) reports the syntax.

# In[13]:


import pyomo.kernel as pyk

def BIMWithBallUncertainty(radius, domain_type=pyk.RealSet):
    
    idxChips = range(len(chips))
    
    m = pyk.block()

    m.x = pyk.variable_list()
    for i in idxChips:
        m.x.append(pyk.variable(lb=0,domain_type=domain_type))
    
    m.profit    = pyk.objective(expr = sum(profits[chips[i]]*m.x[i] for i in idxChips), sense=pyk.maximize)

    m.silicon   = pyk.constraint(expr = m.x[0]  <= 1000)
    m.germanium = pyk.constraint(expr = m.x[1]  <= 1500)
    m.plastic   = pyk.constraint(expr = sum([m.x[i] for i in idxChips]) <= 1750)

    m.y = pyk.variable(lb=0)
    m.w = pyk.variable_list()
    for i in idxChips:
        m.w.append(pyk.variable(lb=0))

    m.copper = pyk.constraint(expr = m.y == 4800 - sum(copper[chips[i]]*m.x[i] for i in idxChips)) 

    m.xtow = pyk.constraint_list()
    for i in idxChips:
        m.xtow.append(pyk.constraint(expr = m.w[i] == radius * m.x[i]))

    from pyomo.core.kernel.conic import quadratic
    m.robust = quadratic(m.y,m.w)

    return m


# Now the optimization problem is nonlinear, but dedicated solvers can leverage the fact that is conic and solve it efficiently. Specifically, `cplex`, `gurobi` and `xpress` support second-order cones. On the other hand, `ipopt` is a generic solver for nonlinear optimization problems.

# In[14]:


radius = 0.05
m = BIMWithBallUncertainty(radius)

results = gurobi.solve(m)
print('Solver: gurobi, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3),"\n")

results = cplex.solve(m)
print('Solver: cplex, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3))

results = xpress.solve(m)
print('Solver: xpress, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3),"\n")

results = ipopt.solve(m)
print('Solver: ipopt, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3))


# The solvers `cplex`, `gurobi` and `xpress` are capable of solving the mixed integer version of the same model: 

# In[15]:


m = BIMWithBallUncertainty(radius, domain_type=pyk.IntegerSet)

results = gurobi.solve(m)
print('Solver: gurobi, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3),"\n")

results = cplex.solve(m)
print('Solver: cplex, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3))

results = xpress.solve(m)
print('Solver: xpress, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print('The optimal solution is x=',[round(pyk.value(m.x[i]),3) for i in range(len(m.x))],'and yields a profit of',round(pyk.value(m.profit),3))


# ### Implementing second-order cones using `pyomo.environ`
# 
# Noting that $\| x \| \leq t$ is for $t \geq 0$ equivalent to $\| x \|^2 \leq t^2$ and knowing that the commercial solvers (`gurobi`, `cplex` and `express`) support convex quadratic inequalities, we can model this variant in `pyomo.environ` as follows. Note that the essential part to make the model convex is having the right hand side nonnegative.

# In[16]:


def BIMWithBallUncertaintyAsSquaredSecondOrderCone(r, domain=pyo.NonNegativeReals):
    
    m = pyo.ConcreteModel('BIM with Ball Uncertainty as SOC')
    
    m.chips = pyo.Set(initialize=chips)
    m.x = pyo.Var(m.chips,within=domain)

    # the nonnegativity of this variable is essential!
    m.y = pyo.Var(within=pyo.NonNegativeReals)

    m.profit    = pyo.Objective(expr = sum([profits[c]*m.x[c] for c in m.chips]), sense=pyo.maximize)

    m.silicon   = pyo.Constraint(expr = m.x['logic']  <= 1000)
    m.germanium = pyo.Constraint(expr = m.x['memory']      <= 1500)
    m.plastic   = pyo.Constraint(expr = sum([m.x[c] for c in m.chips]) <= 1750)
    m.copper    = pyo.Constraint(expr = m.y == 4800 - sum(copper[c]*m.x[c] for c in m.chips)) 
    m.robust    = pyo.Constraint(expr = sum((r*m.x[c])**2 for c in m.chips) <= m.y**2)
    
    return m


# In[17]:


m = BIMWithBallUncertaintyAsSquaredSecondOrderCone(radius)

results = gurobi.solve(m)
print('Solver: gurobi, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}\n')

results = cplex.solve(m)
print('Solver: cplex, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# In[18]:


m = BIMWithBallUncertaintyAsSquaredSecondOrderCone(radius, domain=pyo.NonNegativeIntegers)

results = gurobi.solve(m)
print('Solver: gurobi, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}\n')

results = cplex.solve(m)
print('Solver: cplex, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# Note how the verbose `xpress` solver confirms that the convex quadratic constraint is recognized as conic.

# In[19]:


m = BIMWithBallUncertaintyAsSquaredSecondOrderCone(radius, domain=pyo.NonNegativeIntegers)

results = xpress.solve(m,tee=True)
print('\nSolver: xpress, solver status:',results.solver.status, 'and solver terminal condition:',results.solver.termination_condition)
print(f'The optimal solution is x={[round(pyo.value(m.x[c]),3) for c in m.chips]} and yields a profit of {pyo.value(m.profit):.2f}')


# In[ ]:




