#!/usr/bin/env python
# coding: utf-8

# ```{index} single: Pyomo; variables 
# ```
# ```{index} single: Pyomo; expressions 
# ```
# ```{index} single: Pyomo; objectives 
# ```
# ```{index} single: Pyomo; constraints
# ```
# ```{index} single: Pyomo; decorators
# ```
# ```{index} single: solver; cbc
# ```
# 
# # A Simple Pyomo Model
# 
# Pyomo is an algebraic modeling language embedded within Python for the purpose of implementing applications of mathematical optimization. Pyomo is used to create model object consisting of decision variables, expressions, objective functions, and constraints. Once a model has been specified, Pyomo includes methods to perform various transformations of the model, and then to solve models with a choice of open-source and commercial solvers. Pyomo is open source, not tied to any specific class of mathematical optimization problems, and undergoing continued development with contributed third-party packages.
# 
# This notebook introduces the basic elements of Pyomo for the case study of a small production planning problem. A second notebook demonstrates additional features of Pyomo that make it possible to write "data-driven" applications. Finally, a more complete version of the problem is revisited in Chapter 10 to demonstrate stochastic and robust optimization. 
# 
# This notebook introduces components from the Pyomo library common to most applications:
# 
# * [Variables](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Variables.html)
# * [Expressions](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html)
# * [Objectives](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Objectives.html)
# * [Constraints](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Constraints.html)
# * [SolverFactory](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html)
# 
# This Pyomo model presented below is a direct translation of the mathematics into these basic Pyomo components. For brevity, the problem parameter values are incorporated directly in the model. This generates a satisfactory result for examples with a small number of decision variables and constraints.
# 
# This notebook also introduces the use of Python decorators to designate Pyomo objectives, constraints, and other model components. This is a relatively new feature in Pyomo  available in recent versions. Decorators may be unfamiliar to new users of Python (or current users of Pyomo), but are  worth learning in return for a remarkable gain in the readability of Pyomo models.

# ## Preliminary Step: Install Pyomo and solvers
# 
# We start by verifying the installation of Pyomo and any needed solvers. The following cell downloads a Python module that can check if Pyomo and designated solvers have been previously installed. If no installation is detected, the functions will perform the needed installations. This step needs to be done only once for a given installation of Python on a personal laptop. On Google Colab, however, this step must also be done for each new instance of a Colab session.

# In[1]:


import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Step 1. Import Pyomo
# 
# The first coding step for a new Pyomo model is to import the needed components into the Python environment. Importing  `pyomo.environ` provides the most commonly used components for Pyomo model building. This collection of notebooks uses a consistent convention of using `pyo` as a prefix to objects imported from `pyomo.environ`.

# In[2]:


import pyomo.environ as pyo


# ## Step 2. Create a `ConcreteModel` object
# 
# A Pyomo model is created in a Python workspace with a standard Python variable name. Here and in subsequent notebooks the model object is be created with `pyo.ConcreteModel()`. `pyo.ConcreteModel` is used when the problem data is known at the time the model is created. Pyomo also provides `pyo.AbstractModel()` for creating abstract models where problem data is provided later to create specific model instances. This same effect can be achieved in Python using the "data driven" approach demonstrated throughout this collection of notebooks.
# 
# The following cell creates an instance of ConcreteModel object and stores it in a Python variable named `model`. The name can be any valid Python variable. Generally a short name is desirable since it will be a prefix for every Pyomo variable and constraint.

# In[3]:


# create model with optional problem name
model = pyo.ConcreteModel("Production Planning: Version 1")


# The `.display()` method is a convenient means of displaying and verifying the current contents of a Pyomo model. At this stage the major components of the model are empty.

# In[4]:


#display model
model.display()


# ## Step 3. Create decision variables
# 
# Decision variables are created with the Pyomo `Var()`. The decisions variables can be assigned to any valid Python identifier. Here we assign the decision variables to the model instance using the Python 'dot' notation using a name derived from the mathematical model. 
# 
# `Var()` accepts optional keyword arguments:
# 
# * `bounds` specifies a tuple containing lower and upper bounds. A good modeling practice is to specify any known and fixed bounds on the decision variables as they are created. If one of the bounds is unknown, use `None` as a placeholder.
# 
# * `initialize` specifies initial values for the decision variables. This isn't normally required, but is useful in this tutorial example where we want to display the model as it is being built.
# 
# * `domain` keyword specifies the type of decision variable. By default, the domain is all real numbers including negative and positive values. 
# 
# The use of the optional keywords is shown in the following cell.

# In[5]:


# create decision variables
model.x_M = pyo.Var(bounds=(0, None), initialize=0)
model.x_A = pyo.Var(bounds=(0, 80), initialize=0)
model.x_B = pyo.Var(bounds=(0, 100), initialize=0)

model.y_U = pyo.Var(bounds=(0, 40), initialize=0)
model.y_V = pyo.Var(bounds=(0, None), initialize=0)

# display updated model
model.display()


# ## Step 4. Create expressions
# 
# The next cell creates linear expressions for revenue and cost and assigns them to the model as `model.revenue` and `model.cost`, respectively. These expressions will be used to create the profit objective. The expressions can be printed to verify correctness.

# In[6]:


# create expressions
model.revenue = 270 * model.y_U + 210 * model.y_V
model.cost = 10 * model.x_M + 50 * model.x_A + 40 * model.x_B

# expressions can be printed
print(model.revenue)
print(model.cost)


# ## Step 5. Create objective
# 
# The objective is to maximize profit, where  profit is defined as the difference between revenue and cost.
# 
# The expression to be optimized is defined using the `expr` keyword argument. The type of optimization (minimization or maximization) can be specified using the `sense` keyword argument. For example, to maximize profit, defined as the difference between revenue and cost: 
# 
#     model.profit = pyo.Objective(expr = model.revenue - model.cost, sense = pyo.maximize)
#     
# Recent releases of Pyomo also provide a decorator syntax to create an objective function. Using the @model.Objective(sense = pyo.maximize) decorator:
# 
#     @model.Objective(sense = pyo.maximize)
#     def profit(model):
#         return model.revenue - model.cost
# 
# Decorators change the behavior of a function. In this case, the decorator `model.Objective()` modifies `profit()` to return an expression for the profit. The  keyword `sense` sets the type  objective (`pyo.maximize` or `pyo.minimize`). The decorated function profit() takes the Pyomo model as its first argument and its name is added to the model attributes.
# 
# This alternative syntax allows for improved readability and maintainability of more complex models.  Decorators provide a way to simplify the syntax for creating different objects, such as objectives, expressions, constraints, etc., in optimization applications.  

# In[7]:


@model.Objective(sense=pyo.maximize)
def profit(model):
    return model.revenue - model.cost

model.display()


# ## Step 6. Add constraints
# 
# Constraints are logical relationships between expressions that define the range of feasible solutions in an optimization problem. The logical relationships can be equality (`==`), less-than (`<=`), or greater-than (`>=`). 
# 
# `pyo.Constraint()` is a Pyomo class to creating constraints between expressions. A constraint consists of two expressions separated by one of the logical relationships. The constraint is passed as a keyword argument `expr` to `pyo.Constraint()`. For this application the constraints can be expressed as  
# 
#     model.raw_materials = pyo.Constraint(expr = 10 * model.y_U + 9 * model.y_V <= model.x_M)
#     model.labor_A = pyo.Constraint(expr = 2 * model.y_U + 1 * model.y_V <= model.x_A)
#     model.labor_B = pyo.Constraint(expr = 1 * model.y_U + 1 * model.y_V <= model.x_B)
#     
# A `model.Constraint()` decorator is an alternative syntax. The decorator 'tags' the output of the following function as a constraint. For the present example the decorator system is would be

# In[8]:


@model.Constraint()
def raw_materials(model):
    return 10 * model.y_U + 9 * model.y_V <= model.x_M

@model.Constraint()
def labor_A(model):
    return 2 * model.y_U + 1 * model.y_V <= model.x_A

@model.Constraint()
def labor_B(model):
    return 1 * model.y_U + 1 * model.y_V <= model.x_B

model.pprint()


# ## Step 7. Solve the model
# 
# With the model now fully specified, the next step is to compute a solution. A solver object is created with `SolverFactory`, then applied to the model as shown in the following cell. Here we have chosen to use the open source [COIN-OR Cbc](https://github.com/coin-or/Cbc)  ("COIN-OR branch and cut") solver for mixed integer linear programming. There are suitable solver, such as the open source GNU Linear Programming Kit [GLPK](https://en.wikibooks.org/wiki/GLPK) and commercial solvers such as CPLEX, Gurobi, and Mosek.
# 
# This cell shows how to use create and use a solver. The optional keyword `tee=True` causes the solver to print its output to the output. This can be useful for verifying the model is bdebugging problems

# In[9]:


solver = pyo.SolverFactory("cbc")
results = solver.solve(model, tee=True)


# The solver returns a results object that reports inforomation about the problem, and theti

# ## Step 8. Reporting the solution
# 
# Pyomo provides methods to display the solution to an optimization problem. The ba
# 
# For an overview of other ways to report and visualize the solutions, see also the appendix of [this notebook](../04/gasoline-distribution.ipynb).

# In[10]:


model.__dict__


# In[ ]:





# In[ ]:





# In[11]:


pyo.value(model.x_M)


# In[12]:


pyo.value(model.labor_A)


# In[13]:


pyo.value(model.profit)


# In[14]:


model.pprint()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




