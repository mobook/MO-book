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
# Pyomo is an algebraic modeling language for mathematical optimization that is directly embedded within Python. Pyomo is used to create models consisting of decision variables, expressions, objective functions, and constraints. Pyomo includes methods to perform transformations of models, and then to solve models using a choice of open-source and commercial solvers. Pyomo is open source, not tied to any specific class of mathematical optimization problems, and undergoing continuous development with contributed third-party packages.
# 
# This notebook introduces basic elements of Pyomo for the small [production planning problem](https://mobook.github.io/MO-book/notebooks/01/production-planning.html) introduced in a companion notebook. The following cells introduce components from the Pyomo library common to most applications:
# 
# * [Variables](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Variables.html)
# * [Expressions](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html)
# * [Objectives](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Objectives.html)
# * [Constraints](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Constraints.html)
# * [SolverFactory](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html)
# 
# This Pyomo model presented below is a direct translation of the mathematics into these basic Pyomo components. For brevity, the problem parameter values are incorporated directly in the model. This generates a satisfactory result for examples with a small number of decision variables and constraints. A subsequent notebook will demonstrate additional Pyomo features of that make it possible to write "data-driven" applications.
# 
# This notebook also introduces the use of Python decorators to designate Pyomo objectives, constraints, and other model components. This is a relatively new feature in Pyomo  available in recent versions. Decorators may be unfamiliar to new users of Python (or current users of Pyomo), but are  worth learning in return for a remarkable gain in the readability of Pyomo models.

# ## Preliminary Step: Install Pyomo and solvers
# 
# We start by verifying the installation of Pyomo and any needed solvers. The following cell downloads a Python module that checks iff Pyomo and designated solvers have been previously installed.  The functions performs the needed installations if no prior installation is detected. Installation needs to be done once for each Python environment on a personal laptop. For Google Colab, however, this a new installation must be done for each Google Colab session.

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
# The first step for a new Pyomo model is to import the needed components into the Python environment. The module `pyomo.environ` provides the most commonly used components for Pyomo model building. These  notebooks use a consistent convention of importing `pyomo.environ` with the  `pyo` prefix.

# In[2]:


import pyomo.environ as pyo


# ## Step 2. Create a `ConcreteModel` object
# 
# A Pyomo model can have any standard Python variable name. A model object is created with `pyo.ConcreteModel()` when the problem data is known at the time when the model is constructed. Pyomo also provides `pyo.AbstractModel()` for creating models where the problem data will be provided later to create specific model instances. However, the same effect can be achieved in Python when using the "data driven" approach demonstrated in this collection of notebooks.
# 
# The following cell creates an instance of `ConcreteModel` and stores it in a Python variable named `model`. A short name is generally desirable since it will be a prefix for every Pyomo variable and constraint. `ConcreteModel` accepts an optional string argument to add a title for subsequent reports.

# In[3]:


# create model with optional problem name
model = pyo.ConcreteModel("Production Planning: Version 1")


# The `.display()` method is a convenient means of displaying and verifying the current contents of a Pyomo model. At this stage the major components of the model are empty.

# In[4]:


#display model
model.display()


# ## Step 3. Decision variables
# 
# Decision variables are created with `pyo.Var()`. The decisions variables can be assigned to any valid Python identifier. Here we assign decision variables to the model instance using the Python 'dot' notation. The variable names are chosen to reflect their names in the mathematical model. 
# 
# `pyo.Var()` accepts optional keyword arguments. The most commonly used keyword arguments are:
# 
# * `domain` specifies a set of values for a decision variable. By default the domain is the set of all real number. Other commonly used domains are `pyo.NonNegativeReals`, `pyo.NonNegativeIntegers`, and `pyo.Binary`.
# 
# * `bounds` specifies a tuple containing values for the lower and upper bounds. Good modeling practice specifies any known and fixed bounds on the decision variables. `None` can be used as a placeholder if one of the bounds is unknown. Specifying the bounds as `(0, None)` is equivalent to specifying the domain as `pyo.NonNegativeReals`.
# 
# The use of the optional keywords is shown in the following cell. Displaying the model shows the value of the decision variables are not yet known.

# In[5]:


# create decision variables
model.x_M = pyo.Var(bounds=(0, None))
model.x_A = pyo.Var(bounds=(0, 80))
model.x_B = pyo.Var(bounds=(0, 100))

model.y_U = pyo.Var(bounds=(0, 40))
model.y_V = pyo.Var(bounds=(0, None))

# display updated model
model.display()


# ## Step 4. Expressions
# 
# Pyomo expressions are mathematical formulas that involve the decision variables. In the following code cell, expressions for revenue and cost are created using Pyomo and assigned to `model.revenue` and `model.cost`, respectively. Later on, these expressions will be utilized to establish the profit objective.

# In[6]:


# create expressions
model.revenue = 270 * model.y_U + 210 * model.y_V
model.cost = 10 * model.x_M + 50 * model.x_A + 40 * model.x_B

# expressions can be printed
print(model.revenue)
print(model.cost)


# ## Step 5. Objective
# 
# The objective for this example is to maximize profit given by the difference between revenue and cost. There are two ways this objective can be specified in Pyomo.
# 
# The first method is to use `pyo.Objective()` where the expression to be optimized is assigned with the `expr` keyword and the type of objective is assigned with the `sense` keyword.
# 
#     model.profit = pyo.Objective(expr = model.revenue - model.cost, sense = pyo.maximize)
#     
# Recent releases of Pyomo provide a second method that uses Python [decorators](https://peps.python.org/pep-0318/) to specify an objective. Using a decorator, the same objective is written as
# 
#     @model.Objective(sense = pyo.maximize)
#     def profit(model):
#         return model.revenue - model.cost
# 
# Python decorators are a way to modify the behavior of a function. In this case, the decorator `@model.Objective()` modifies the behavior of the profit() function to return an expression for the profit. The sense keyword is used to set the type of objective, which can either be to maximize or minimize the objective function. The profit() function, after being decorated, takes the Pyomo model as its first argument and adds its name to the model attributes.
# 
# In essence, decorators act as tags that modify the behavior of a function. In more complex models, decorators can improve the readability and maintainability of the code. They simplify the syntax for creating other Pyomo objects expressions, constraints, and other optimization-related elements.

# In[7]:


@model.Objective(sense=pyo.maximize)
def profit(model):
    return model.revenue - model.cost

model.display()


# ## Step 6. Add constraints
# 
# Constraints are logical relationships between expressions that define the range of feasible solutions in an optimization problem. The logical relationships can be equality (`==`), less-than (`<=`), or greater-than (`>=`). 
# 
# `pyo.Constraint()` is a Pyomo class to creating constraints between expressions. A constraint consists of two expressions separated by one of the logical relationships. The constraint is passed as a keyword argument `expr` to `pyo.Constraint()`. For this application the constraints are expressed as  
# 
#     model.raw_materials = pyo.Constraint(expr = 10 * model.y_U + 9 * model.y_V <= model.x_M)
#     model.labor_A = pyo.Constraint(expr = 2 * model.y_U + 1 * model.y_V <= model.x_A)
#     model.labor_B = pyo.Constraint(expr = 1 * model.y_U + 1 * model.y_V <= model.x_B)
#     
# A `@model.Constraint()` decorator provides an alternative syntax. The decorator 'tags' the output of the following function as a constraint. For the present example, the constraints are expressed with decorators as follows:

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
# With the model now fully specified, the next step is to compute a solution. A solver object is created with `SolverFactory` then applied to the model as shown in the following cell. Here we have chosen to use the open source [COIN-OR Cbc](https://github.com/coin-or/Cbc)  ("COIN-OR branch and cut") solver for mixed integer linear programming. There are other suitable solvers such as the open source GNU Linear Programming Kit [GLPK](https://en.wikibooks.org/wiki/GLPK), or commercial solvers such as CPLEX, Gurobi, and Mosek.
# 
# The optional keyword `tee=True` causes the solver to print its output to the output. This can be useful for debugging problems with the model.

# In[9]:


solver = pyo.SolverFactory("cbc")
results = solver.solve(model, tee=True)


# ## Step 8. Reporting the solution
# 
# The final step in any application is to report the solution in a format suitable for the application. For this example our goal is report the solutions as a tabular and a simple graphic.
# 
# For an overview of other ways to report and visualize the solutions, see also the appendix of [this notebook](../04/gasoline-distribution.ipynb).

# In[10]:


import pandas as pd

production = {"U": pyo.value(model.y_U),
              "V": pyo.value(model.y_V)}
           
pd.Series(production)


# In[11]:


print(f"Production Planning")


# In[ ]:





# In[12]:


pyo.value(model.x_M)


# In[13]:


pyo.value(model.labor_A)


# In[14]:


pyo.value(model.profit)


# In[15]:


model.pprint()


# In[ ]:




