#!/usr/bin/env python
# coding: utf-8

# ```{index} single: Pyomo; variables 
# ```
# ```{index} single: Pyomo; expressions 
# ```
# ```{index} single: Pyomo; sets 
# ```
# ```{index} single: Pyomo; decorators
# ```
# ```{index} single: solver; cbc
# ```
# 
# # A Data-Driven Pyomo Model
# 
# Version 2 of the model introduces the use of sets, parameters, indexed variables, and indexed constraints. These additional components of the Pyomo library are essential to building scalable and maintainable models for more complex applications. 
# 
# * [Sets](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Sets.html)
# * [Parameters](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Parameters.html)
# 
# 

# ## Data Representation
# 
# Choosing an organized and complete representation of the problem data is a productive starting point for creating Pyomo models. In this case.  the data consists of (1) numbers describing the price and demand for products, (2) numbers for the cost and availability of resources needed to produce the products, and (3) numbers describing the amount of resources needed to produce each unit of every product. This suggests three tables to represent the problem data.
# 
# Here we use nested dictionaries as containers for the problem data, and use Pandas DataFrames to display the data. For the product data, the dictionary keys are the names of each product. For each product there is an associated dictionary containing the price and market demand. `None` is used as a placeholder if there is no limit on market demand.

# In[1]:


import pandas as pd

product_data = {
    "U": {"price": 270, "demand": 40},
    "V": {"price": 210, "demand": None},
}

pd.DataFrame(product_data)


# A nested dictionary is also used to represent data on the the available resources. The dictionary keys are the names of each resource. For each resource there is an associated dictionary indicating the price and units of the resource available. `None` is used as a placeholder if there is no apriori limit on the available inventory.

# resource_data = {
#     "M": {"price": 10, "available": None},
#     "labor A": {"price": 50, "available": 100},
#     "labor B": {"price": 40, "available":  80},
# }
# 
# pd.DataFrame(resource_data)

# The third nested dictionary provides a table of data shows the amount of each resource needed to produce one unit of each product. Here the primary keys are the product names and the secondary keys are the resources.

# In[2]:


process_data = {
    "U": {"M": 10, "labor A": 2, "labor B": 1},
    "V": {"M":  9, "labor A": 1, "labor B": 1},
}

pd.DataFrame(process_data)


# ## Pyomo Model

# ### Step 0.Install Pyomo and solvers
# 
# Before going further, the first step is to install the Pyomo library and any solvers that may be used to compute numerical solutions. The following cell downloads a Python module that will check for (and if necessary install) Pyomo and a linear solver on Google Colab and most laptops.

# In[3]:


import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ### Step 1. Import Pyomo
# 
# The first step in creating a Pyomo model is to import the needed components from the Pyomo library into the Python environment. Importing  `pyomo.environ` will provides the most commonly used components for Pyomo model building. 
# 
# This collection of notebooks uses a consistent convention of assigning a `pyo` prefix to objects imported from `pyomo.environ`.

# In[21]:


import pyomo.environ as pyo


# ### Step 2. Create a `ConcreteModel` object
# 
# A Pyomo model is stored in a Python workspace with a standard Python variable name. The model object can be created with either `pyo.ConcreteModel()` or `pyo.AbstractModel()`. `pyo.ConcreteModel` is used when the problem data is known at the time the model is created, which is the case here. `pyo,AbstractModel` is useful for situations where the model is created before the model data is known. 
# 
# The following cell creates a ConcreteModel object and stores it in a Python variable named `model`. The name can be any valid Python variable, but keep in mind the name will be a prefix for every variable and constraint. 

# In[22]:


model = pyo.ConcreteModel("Production Planning: Version 1")


# The `.display()` method is a convenient means of displaying the current contents of a Pyomo model.

# In[23]:


model.display()


# ### Step 3. Create sets of products and resources
# 
# The production planning problem is given 

# ### Step 3. Create decision variables
# 
# Decision variables are created with the Pyomo `Var()` class. Variables are assigned to attributes of the model object using the Python 'dot' notation. `Var()` accepts optional keyword arguments. 
# 
# * The optional `bounds` keyword specifies a tuple containing lower and upper bounds. A good modeling practice is to specify any known and fixed bounds on the decision variables as they are created. If one of the bounds is unknown, use `None` as a placeholder.
# 
# * The `initialize` keyword specifies initial values for the decision variables. This isn't normally required, but is useful in this tutorial example where we want to display the model as it is being built.
# 
# * The optional `domain` keyword specifies the type of decision variable. By default, the domain is all real numbers including negative and positive values. 

# In[24]:


model.x_M = pyo.Var(bounds=(0, None), initialize=0)
model.x_A = pyo.Var(bounds=(0, 80), initialize=0)
model.x_B = pyo.Var(bounds=(0, 100), initialize=0)

model.y_U = pyo.Var(bounds=(0, 40), initialize=0)
model.y_V = pyo.Var(bounds=(0, None), initialize=0)

model.display()


# ### Step 4. Create expressions
# 
# The difference between revenue and expense is equal to the gross profit. The following cell creates linear expressions for revenue and expense which are then assigned the expressions to attributes called `model.revenue` and `model.expense`. We can print the expressions to verify correctness.

# In[25]:


model.revenue = 270 * model.y_U + 210 * model.y_V
model.expense = 10 * model.x_M + 50 * model.x_A + 40 * model.x_B

print(model.revenue)
print(model.expense)


# ### Step 5. Create objective
# 
# The objective is to maximize gross profit, where gross profit is defined as the difference between revenue and expense.
# 
# The Pyomo class `Objective` creates the objective for a Pyomo model. The required keyword argument `expr` specifies the expression to be optimized. The optional keyword argument `sense` specifies whether it is a minimization or maximization problem. For clarity, it is good practice is to always include the `sense` keyword argument.
# 
#     model.profit = pyo.Objective(expr = model.revenue - model.expense, sense = pyo.maximize)
#     
# An alternative way to declare an objective is available in more recent releases of Pyomo. This approach uses a Python decorator to 'tag' function that returns a Pyomo expression as the objective function. 
# 
#     @model.Objective(sense = pyo.maximize)
#     def profit(model):
#         return model.revenue - model.expense
# 
# Decorators are a feature of Python that change the behavior of a function without altering the function itself. The Pyomo library provides decorators to declare objectives, expression, constraints, and other objects for optimization applications. Decorators improve the readability and maintainability of more complex models. Without going into the details of how decorators are implemented, one can think of them as a means for tagging functions for the purpose of creating model objects.
# 
# When using Pyomo decorators, the name of the function will create a model attribute. The function to be tagged with a Pyomo decorator must have the model object as the first argument. The decorator itself may have arguments. For example, the `sense` keyword is used to set the sense of an objective.
# 

# In[121]:


@model.Objective(sense=pyo.maximize)
def profit(model):
    return model.revenue - model.expense

model.display()


# **Step 5. Add constraints.**
# 
# Constraints are logical relationships between expressions that define the range of feasible solutions in an optimization problem.. The logical relationships can be `==`, `<=`, or `>=`. 
# 
# `pyo.Constraint()` is a Pyomo library component to creating scalar constraints. A constraint consists of a logical relationship between expressions is passed as a keyword argument `expr` to `pyo.Constraint()`. For this application, the constraints are expressed as  
# 
#     model.raw_materials = pyo.Constraint(expr = 10 * model.y_U + 9 * model.y_V <= model.x_M)
#     model.labor_A = pyo.Constraint(expr = 2 * model.y_U + 1 * model.y_V <= model.x_A)
#     model.labor_B = pyo.Constraint(expr = 1 * model.y_U + 1 * model.y_V <= model.x_B)
#     
# Alternatively, the decorator syntax for constraints provides a readable and maintainable means to express more complex constraints. For the present example, the constraints would be writte
# 
#     @model.Constraint()
#     def raw_materials(model):
#         return 10 * model.y_U + 9 * model.y_V <= model.x_M
# 
#     @model.Constraint()
#     def labor_A(model):
#         return 2 * model.y_U + 1 * model.y_V <= model.x_A
# 
#     @model.Constraint()
#     def labor_B(model):
#         return 2 * model.y_U + 1 * model.y_V <= model.x_A
# 
# These are demonstrated in the following cell.

# In[104]:


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


# **Step 6. Solve the model.**
# 
# With the model now fully specified, the next step is to compute a solution. This is done by creating a solver object using `SolverFactory`, then applying the solver to the model, as shown in the following cell.
# 
# EXPAND DISCUSSION REGARDING SOLVERS.

# In[105]:


solver = pyo.SolverFactory("cbc")
solver.solve(model)

model.pprint()


# **Step 7. Reporting the solution.**

# **Enhancement 1: Create sets of products and resources**

# In[80]:


model = pyo.ConcreteModel("Production Planning: Version 2")

model.PRODUCTS = pyo.Set(initialize=product_data.keys())
model.RESOURCES = pyo.Set(initialize=resource_data.keys())

def x_bounds(model, r):
    return(0, resource_data[r]["available"])
model.x = pyo.Var(model.RESOURCES, bounds=x_bounds)

def y_bounds(model, p):
    return(0, product_data[p]["demand"])
model.y = pyo.Var(model.PRODUCTS, bounds=y_bounds)

@model.Objective(sense=pyo.maximize)
def profit(model):
    model.expense = sum(resource_data[r]["price"] * model.x[r] for r in model.RESOURCES)
    model.revenue = sum(product_data[p]["price"] * model.y[p] for p in model.PRODUCTS)
    return model.revenue - model.expense

@model.Constraint(model.RESOURCES)
def resource(model, r):
    return sum(process_data[p][r] * model.y[p] for p in model.PRODUCTS) <= model.x[r]

solver = pyo.SolverFactory('cbc')
solver.solve(model)
    
model.display()


# In[ ]:





# In[ ]:




