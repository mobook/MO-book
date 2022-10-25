#!/usr/bin/env python
# coding: utf-8

# # Companion notebook to EF's training on optimization with data uncertainty
# 
# Sander Vlot & Joaquim Gromicho, 2021
# 
# ---
#  > During this course we make use of Jupyter notebooks hosted by [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). 
#  The usage of this platform is allowed by ORTEC for **educational and personal experimentation only**. May you consider using it for a project please consult the IT department.
#  Notebooks deployed on `colab` require neither python nor other dependencies to be installed on your own machine, you only need a browser (preferably `chrome`) and you may also need a google account if you want to execute them. 
#  
# ---
# 
# This notebook has been setup for `colab`. Check [the pyomo cookbook](https://jckantor.github.io/ND-Pyomo-Cookbook/) and in particular [the explanation on how to get pyomo and the solvers on colab](https://jckantor.github.io/ND-Pyomo-Cookbook/01.02-Running-Pyomo-on-Google-Colab.html).
# 
# May you want to use on your own python distribution, then you should care for  the installation of the required packages and subsidiary applications. 
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


# In[2]:


get_ipython().system('pyomo help --solvers')


# In[3]:


import pyomo.environ as pyo
cbc    = pyo.SolverFactory('cbc')
ipopt  = pyo.SolverFactory('ipopt')
cplex  = pyo.SolverFactory('cplex_direct')
gurobi = pyo.SolverFactory('gurobi_direct')
xpress = pyo.SolverFactory('xpress_direct')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Recall the Alice's production planning model
# 
# Alice owns a company that produces trophies for
# * football
#  * wood base, engraved plaque, brass football on top
#  * €12 profit and uses 4 dm of wood
# * golf
#  * wood base, engraved plaque, golf ball on top
#  * €9 profit and uses 2 dm of wood
# 
# Alice’s current stock of raw materials
# * 1000 footballs
# * 1500 golf balls
# * 1750 plaques
# * 480 m (4800 dm) of wood
# 
# > Alice wonders what the optimal production plan should be, in other words: how many football and how many golf trophies should Alice produce to maximize his profit while respecting the availability of raw materials?
# 
# ***
# 
# The following model __maximizes__ Alice's profit by deciding the number of $x_1$ football and $x_2$ golf trophies to produce.
# 
# $$
# \begin{array}{rrcrcl}
# \max    & 12x_1 & + & 9x_2               \\
# s.t.    &   x_1 &   &      & \leq & 1000 \\
#         &       &   &  x_2 & \leq & 1500 \\
#         &   x_1 & + &  x_2 & \leq & 1750 \\
#         &  4x_1 & + & 2x_2 & \leq & 4800 \\
#         &   x_1 & , &  x_2 & \geq & 0    \\
# \end{array}
# $$

# In[5]:


trophies = [ 'Football', 'Golf' ]
profits  = { 'Football' : 12, 'Golf' :  9 }
wood     = { 'Football' :  4, 'Golf' :  2 }

Alice = pyo.ConcreteModel('Alice')

Alice.x = pyo.Var(trophies,within=pyo.NonNegativeReals)

Alice.profit    = pyo.Objective(expr = sum([profits[t]*Alice.x[t] for t in trophies]), sense=pyo.maximize)

Alice.footballs = pyo.Constraint(expr = Alice.x['Football']  <= 1000)
Alice.golfBalls = pyo.Constraint(expr = Alice.x['Golf']      <= 1500)
Alice.plaques   = pyo.Constraint(expr = sum([Alice.x[t] for t in trophies]) <= 1750)
Alice.wood      = pyo.Constraint(expr = sum(wood[t]*Alice.x[t] for t in trophies) <= 4800 ) 


# In[6]:


get_ipython().run_line_magic('time', 'results = cbc.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )

print(Alice.profit.expr())
print([Alice.x[t].value for t in trophies])

Alice.display()


# In[7]:


Alice.pprint()


# In[8]:


def ShowModelComponents( model ):
    for v in model.component_objects(pyo.Var, active=True):
        print ("Variable  ",v)
        varobject = getattr(model, str(v))
        for index in varobject:
            print ("     ",index, varobject[index].value)
    for o in model.component_objects(pyo.Objective, active=True):
        print ("Objective ",o)
        varobject = getattr(model, str(o))
        for index in varobject:
            print ("    ",index, varobject[index].expr())
    for c in model.component_objects(pyo.Constraint, active=True):
        print ("Constraint",c)
        varobject = getattr(model, str(c))
        for index in varobject:
            print ("     ",index, varobject[index].uslack())


# In[9]:


ShowModelComponents( Alice )


# In[10]:


def ShowDuals( model ):
    import fractions
    # display all duals
    print ("Duals")
    for c in model.component_objects(pyo.Constraint, active=True):
        print ("Constraint ",c)
        for index in c:
            print ("      ", index, str(fractions.Fraction(model.dual[c[index]])))


# In[11]:


Alice.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

get_ipython().run_line_magic('time', 'results = cbc.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )


# In[12]:


ShowDuals( Alice )


# In[13]:


def JustSolution( model ):
  return [ pyo.value(model.profit) ] + [ pyo.value(model.x[i]) for i in trophies ]


# In[14]:


JustSolution( Alice )


# ## Betty: a tiny bit of data science... 
# 
# 
# We start by simulating two samples of observed wood lengths for `f` football trophies and `g` golf trophies.

# In[15]:


import numpy as np 
np.random.seed(2021)

n = 2000

f = np.random.lognormal(np.log(4.), .005, n)
g = np.random.lognormal(np.log(2.), .005, n)


# In[16]:


print(f)
print(g)


# In[17]:


print( min(f), max(f), min(g), max(g) )


# In[18]:


import matplotlib.pyplot as plt

plt.plot( f, '.' )
plt.plot( g, '.' )
plt.show()


# What is the consequence of uncertainty? 
# We compare the cumulative lengths with the nominal ones.

# In[19]:


cs = np.cumsum(g) - np.cumsum( [2]*len(g) )
plt.plot(cs)
plt.show()


# In[20]:


plt.pie( [ sum( cs > 0 ), sum( cs <= 0 ) ], labels = [ 'trouble!', 'ok' ], autopct='%1.1f%%', shadow=True, startangle=90, colors=[ 'red', 'green' ])
plt.show()


# A very simple and somehow naïf uncertainty region can be taken as the observed minimal box around the data.

# In[21]:


import matplotlib.patches as patches

plt.figure()
plt.plot( f, g, '.' )
currentAxis = plt.gca()
currentAxis.add_patch(patches.Rectangle((min(f), min(g)), max(f)-min(f), max(g)-min(g),fill=False,color='r'))
plt.show()

print( min(f), max(f), min(g), max(g) )


# ## Caroline's robust model for box uncertainty in wood consumption
# 
# Suppose now that Alice notices that not _exactly_ 4 and 2 dm of wood are used, but some fluctuations are observed.
# Alice wants to be __sure__ that her model does not violate the wood constraint, therefore the following should hold:
# 
# $$
# \begin{array}{rrcrcl}
# \max    &  12 x_1 & + &   9 x_2               \\
# s.t.    &     x_1 &   &         & \leq & 1000 \\
#         &         &   &     x_2 & \leq & 1500 \\
#         &     x_1 & + &     x_2 & \leq & 1750 \\
#         & a_1 x_1 & + & a_2 x_2 & \leq & 4800 & \forall \ell \leq a \leq u \\
#         &     x_1 & , &     x_2 & \geq & 0    \\
# \end{array}
# $$
# 
# ***
# 
# A bit of linear duality (or even better: an introduction to robust optimization!) helps Alice how to deal with the above model that has an infinite number of constraints.
# The first thing to notice is that the wood consumption is modeled by constraints that are equivalent to bounding the following optimization problem:
#     
# $$
# \begin{array}{rrr}
# \max    & x_1 a_1 + x_2 a_2 & \leq 4800 \\
# s.t.    & \ell \leq a \leq u 
# \end{array}
# $$
# 
# Or
# 
# $$
# \begin{array}{rrr}
# \max    & x_1 a_1 + x_2 a_2 & \leq 4800 \\
# s.t.    & a \leq u \\
#         & -a \leq -\ell 
# \end{array}
# $$
# 
# Now we use linear duality to realize that the above is equivalent to:
#  
# $$
# \begin{array}{rrr}
# \min    & u y  - \ell w & \leq 4800 \\
# s.t.    & y - w = x \\
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
# The only thing we need to do is add variables and constraints to Alice's model.

# # A model in `pyomo`

# In[22]:


def AliceWithBoxUncertainty( lower, upper, domain=pyo.NonNegativeReals ):

    Alice = pyo.ConcreteModel('AliceBox')

    Alice.x = pyo.Var(trophies,within=domain)

    Alice.profit    = pyo.Objective(expr = sum([profits[t]*Alice.x[t] for t in trophies]), sense=pyo.maximize)

    Alice.footballs = pyo.Constraint(expr = Alice.x['Football']  <= 1000)
    Alice.golfBalls = pyo.Constraint(expr = Alice.x['Golf']      <= 1500)
    Alice.plaques   = pyo.Constraint(expr = sum([Alice.x[t] for t in trophies]) <= 1750)

    Alice.y = pyo.Var(trophies,domain=pyo.NonNegativeReals)
    Alice.w = pyo.Var(trophies,domain=pyo.NonNegativeReals)

    Alice.robustWood = pyo.Constraint(expr = sum([upper[t]*Alice.y[t] - lower[t]*Alice.w[t] for t in trophies]) <= 4800)

    def PerVariable( model, t ):
        return model.x[t] == model.y[t] - model.w[t] 
    
    Alice.perVariable = pyo.Constraint(trophies,rule=PerVariable)

    return Alice


# In[23]:


lower = upper = {}
lower['Football'] = min(f)
upper['Football'] = max(f)
lower['Golf'] = min(g)
upper['Golf'] = max(g)

Alice = AliceWithBoxUncertainty( lower, upper, domain=pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', 'results = cbc.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )

JustSolution( Alice )


# In[24]:


# you can play with the amount of uncertainty. 
# In particular, if below you make delta equal to 0 you obtain the same result als the  nominal model.
delta = 0.05

def AliceWithSymmetricalBoxUncertainty( delta, domain=pyo.NonNegativeIntegers ):
    lower = { trophy : wood[trophy] - delta for trophy in wood }
    upper = { trophy : wood[trophy] + delta for trophy in wood }
    return AliceWithBoxUncertainty( lower, upper, domain=domain )

Alice = AliceWithSymmetricalBoxUncertainty( delta )
get_ipython().run_line_magic('time', 'results = cbc.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )

JustSolution( Alice )


# # Integer optimization
# 
# Alice's model gave integer solutions, but not the robust version. 
# If we need integer solutions then we should impose that to the nature of the variables, which in this case of _box uncertainty_ is easy to do since the model remains linear, although it will be mixed integer. 

# In[25]:


Alice = AliceWithBoxUncertainty( lower, upper, domain=pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', 'results = cbc.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )

JustSolution( Alice )


# In[26]:


import pandas
df = pandas.DataFrame()
for delta in np.linspace(0,.5,21):
  Alice = AliceWithSymmetricalBoxUncertainty( delta, domain=pyo.NonNegativeIntegers )
  cbc.solve(Alice)
  results = JustSolution( Alice )
  df.at[delta,'value']     = results[0]
  df.at[delta,trophies[0]] = results[1]
  df.at[delta,trophies[1]] = results[2]
df


# In[27]:


df.plot()


# In[28]:


df[['Football','Golf']].plot()


# # Cardinality constrained uncertainty
# 
# Each $a_j$ may deviate by at most $\pm \delta_j$ from the nominal value $\bar{a}_j$ bun no more than $\Gamma$ will actually deviate.
# 
# $$
# \begin{array}{rrcrcl}
# \max    &  12 x_1 & + &   9 x_2               \\
# s.t.    &     x_1 &   &         & \leq & 1000 \\
#         &         &   &     x_2 & \leq & 1500 \\
#         &     x_1 & + &     x_2 & \leq & 1750 \\
#         & a_1 x_1 & + & a_2 x_2 & \leq & 4800 & \forall a,y : a_j=\bar{a}_j+\delta_jy_j, \|y\|_\infty \leq 1, \|y\|_1\leq \Gamma \\
#         &     x_1 & , &     x_2 & \geq & 0    \\
# \end{array}
# $$
# 
# As we have seen on the previous lecture, Lagrange duality yields the following modification to the problem as equivalent to the robust model stated above:
# 
# $$
# \begin{array}{rrcrcrcrcrcrcl}
# \max    &  12 x_1 & + &   9 x_2               \\
# s.t.    &     x_1 &   &         & & & & & & & \leq & 1000 \\
#         &         &   &     x_2 & & & & & & & \leq & 1500 \\
#         &     x_1 & + &     x_2 & & & & & & & \leq & 1750 \\
#         & a_1 x_1 & + & a_2 x_2 & + & \lambda\Gamma & + & z_1 & + & z_2 & \leq & 4800 \\
#         &-d_1 x_1 &   &         & + & \lambda & + & z_1 &   &     & \geq & 0 \\
#         &         &   &-d_2 x_2 & + & \lambda &   &     & + & z_2 & \geq & 0 \\
#         &     x_1 & , &     x_2 & , & \lambda & , & z_1 & , & z_2 & \geq & 0    \\
# \end{array}
# $$

# In[29]:


def AliceWithGammaUncertainty( delta, gamma, domain=pyo.NonNegativeReals ):
    Alice = pyo.ConcreteModel('AliceGamma')

    Alice.x = pyo.Var(trophies,within=domain)

    Alice.profit    = pyo.Objective(expr = sum([profits[t]*Alice.x[t] for t in trophies]), sense=pyo.maximize)

    Alice.footballs = pyo.Constraint(expr = Alice.x['Football']  <= 1000)
    Alice.golfBalls = pyo.Constraint(expr = Alice.x['Golf']      <= 1500)
    Alice.plaques   = pyo.Constraint(expr = sum([Alice.x[t] for t in trophies]) <= 1750)

    Alice.z   = pyo.Var(trophies,domain=pyo.NonNegativeReals)
    Alice.lam = pyo.Var(domain=pyo.NonNegativeReals)

    Alice.robustWood = pyo.Constraint( \
     expr = sum([wood[t]*Alice.x[t] for t in trophies]) \
          + gamma * Alice.lam \
          + sum(Alice.z[t] for t in trophies) <= 4800)

    def up_rule( model, t ):
        return model.z[t] >=  delta * model.x[t] - model.lam
    def down_rule( model, t ):
        return model.z[t] >= -delta * model.x[t] - model.lam
    
    Alice.up   = pyo.Constraint(trophies,rule=up_rule)
    Alice.down = pyo.Constraint(trophies,rule=down_rule)

    return Alice


# In[30]:


Alice = AliceWithGammaUncertainty( 0.01, 2, domain=pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', 'results = cbc.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )
JustSolution(Alice)


# # Ball uncertainty
# 
# As [the documentation](https://pyomo.readthedocs.io/en/stable/library_reference/kernel/conic.html) says a conic constraint is expressed in 'pyomo' in simple variables.
# 
# This [table](https://pyomo.readthedocs.io/en/stable/library_reference/kernel/syntax_comparison.html) is very useful. 
# 
# A straightforward remodulation leads to that:
# 
# $$ 
#   a_1x_1+a_2x_2 + \Omega \|x\| \leq 4800
# $$
# 
# $$
#   \Omega \|x\| \leq 4800 - a_1x_1 - a_2x_2
# $$
# $$
#   \|\Omega x\| \leq 4800 - a_1x_1 - a_2x_2
# $$
# 
# By defining $y = 4800 - a_1x_1 - a_2x_2$ we may write:
# $$
#   \Omega^2 \|x\|^2 \leq y^2
# $$
# 
# $$
#   (\Omega x_1)^2 + (\Omega x_2)^2 \leq y^2
# $$
# 
# $$
#   \|w\|^2 \leq y^2
# $$
# 
# with $w = \Omega x$.

# ## Documentation says that we need to use the kernel now

# In[31]:


import pyomo.kernel as pyk

def AliceWithBallUncertainty( omega, domain_type=pyk.RealSet ):
    
    idxTrophies = range( len(trophies) )
    
    Alice = pyk.block()

    Alice.x = pyk.variable_list()
    for i in idxTrophies:
        Alice.x.append( pyk.variable(lb=0,domain_type=domain_type) )
    
    Alice.profit    = pyk.objective(expr = sum(profits[trophies[i]]*Alice.x[i] for i in idxTrophies), sense=pyk.maximize)

    Alice.footballs = pyk.constraint(expr = Alice.x[0]  <= 1000)
    Alice.golfBalls = pyk.constraint(expr = Alice.x[1]  <= 1500)
    Alice.plaques   = pyk.constraint(expr = sum([Alice.x[i] for i in idxTrophies]) <= 1750)

    Alice.y = pyk.variable(lb=0)
    Alice.w = pyk.variable_list()
    for i in idxTrophies:
        Alice.w.append( pyk.variable(lb=0) )

    Alice.wood = pyk.constraint( expr = Alice.y == 4800 - sum(wood[trophies[i]]*Alice.x[i] for i in idxTrophies) ) 

    Alice.xtow = pyk.constraint_list()
    for i in idxTrophies:
        Alice.xtow.append( pyk.constraint( expr = Alice.w[i] == omega * Alice.x[i] ) )

    from pyomo.core.kernel.conic import quadratic
    Alice.robust = quadratic(Alice.y,Alice.w)

    return Alice


# ## Now the problem is nonlinear

# In[32]:


Alice = AliceWithBallUncertainty( 0.1 )

get_ipython().run_line_magic('time', 'results = ipopt.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )
print(pyk.value(Alice.profit))
print( [pyk.value(Alice.x[i]) for i in range(len(Alice.x))] )


# ## But `cplex`, `gurobi` and `xpress` support second order cones

# In[33]:


conicsolver = gurobi


# In[34]:


get_ipython().run_line_magic('time', 'results = conicsolver.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )
print(pyk.value(Alice.profit))
print( [pyk.value(Alice.x[i]) for i in range(len(Alice.x))] )


# ## And therefore we can also have mixed integer models 

# In[35]:


Alice = AliceWithBallUncertainty( 0.1, domain_type=pyk.IntegerSet )

get_ipython().run_line_magic('time', 'results = conicsolver.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )
print(pyk.value(Alice.profit))
print( [pyk.value(Alice.x[i]) for i in range(len(Alice.x))] )


# ## Final note: maybe useful to recall that in python you can always ask for help...

# In[36]:


help(Alice.y)


# In[37]:


help(Alice.robust)


# # How to bring second order cones into the `pyomo.environ`
# 
# Noting that $\| x \| \leq t$ is for $t \geq 0$ equivalent to $\| x \|^2 \leq t^2$ and knowing that the commercial solvers (`gurobi`, `cplex` and `express`) support convex quadratic inequalities, we can model this variant in `pyomo.environ` as follows.
# 
# Note that the essential part to make the model convex is having the rght hand side nonnegative.

# In[38]:


def AliceWithBallUncertaintyAsSquaredSecondOrderCone(omega,domain=pyo.NonNegativeReals):
  Alice = pyo.ConcreteModel('Alice')

  Alice.x = pyo.Var(trophies,within=domain)

  # the nonegativity of this variable is essential!
  Alice.y = pyo.Var(within=pyo.NonNegativeReals)

  Alice.profit    = pyo.Objective(expr = sum([profits[t]*Alice.x[t] for t in trophies]), sense=pyo.maximize)

  Alice.footballs = pyo.Constraint(expr = Alice.x['Football']  <= 1000)
  Alice.golfBalls = pyo.Constraint(expr = Alice.x['Golf']      <= 1500)
  Alice.plaques   = pyo.Constraint(expr = sum([Alice.x[t] for t in trophies]) <= 1750)
  Alice.wood      = pyo.Constraint(expr = Alice.y == 4800 - sum(wood[t]*Alice.x[t] for t in trophies) ) 
  Alice.robust    = pyo.Constraint(expr = sum((omega*Alice.x[t])**2 for t in trophies) <= Alice.y**2) 
  return Alice


# In[39]:


Alice = AliceWithBallUncertaintyAsSquaredSecondOrderCone( 0.1, domain=pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', 'results = cplex.solve(Alice)')
print(results.solver.status, results.solver.termination_condition )
JustSolution(Alice)


# Note how the verbose `xpress` solver confirms that the convex quadratic constraint is recognized as conic.

# In[40]:


Alice = AliceWithBallUncertaintyAsSquaredSecondOrderCone( 0.1, domain=pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', 'results = xpress.solve(Alice,tee=True)')
print(results.solver.status, results.solver.termination_condition )
JustSolution(Alice)


# In[ ]:




