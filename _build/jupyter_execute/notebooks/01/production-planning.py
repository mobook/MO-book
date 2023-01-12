#!/usr/bin/env python
# coding: utf-8

# # Pyomo Tutorial Example: Production Planning
# 
# Pyomo is an algebraic modeling language embedded within Python for creating mathematical optimization applications. Like other modeling languages, with Pyomo the user can create objects representing decision variables, and expressions involving the decision variables. The expressions can be used to create objective functions, or combined in logical relationships to create constraints. Pyomo provides methods to transform and solve models numerically with a choice of open-source and commercial solvers. 
# 
# The purpose of this notebook is introduce the basics of Pyomo by solving a small production planning problem. A more complete version of the problem is revisited in Chapter 10 to demonstrate stochastic and robust optimization. This notebook introduces components from the Pyomo library that are present in most applications:
# 
# * [Sets](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Sets.html)
# * [Parameters](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Parameters.html)
# * [Variables](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Variables.html)
# * [Objectives](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Objectives.html)
# * [Constraints](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Constraints.html)
# * [Expressions](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html)
# 
# The notebook begins with a statement of the problem and the development of a mathematical model. The mathematical model is then translated into several different versions of a Pyomo model. 
# 
# Version 1 of the model is a direct translation of the mathematics into corresponding Pyomo components. This generates a satisfactory solution and would be enough for examples involving a small number of decision variables and constraints. This version of the model provides a tutorial introduction to the use of variables, expressions, objectives, and constraints.
# 
# Version 2 of the model introduces the use of sets, parameters, indexed variables, and indexed constraints. These additional components of the Pyomo library are essential to building scalable and maintainable models for more complex applications. 
# 
# One feature of these notebooks is the use of Python decorators to designate Pyomo objectives, constraints, and other model components. This is a relatively new and, as of the date of this writing, a poorly documented feature of recent versions of Pyomo. Decorators are a less commonly used feature of Python that may be unfamiliar to new users, but are well worth the effort of learning for a remarkable gain in the readability and maintenance of Pyomo models.
# 
# So let's get started.

# ## A Production Planning Problem
# 
# ### Problem Statement
# 
# A small company has developed two versions of a new product. Each version of the product is made from the same raw material that costs 10€ per gram, and requires two different types of specialized labor to produce. 
# 
# $U$ is the higher priced version of the product. $U$ sells for  270€ per unit and requires 10 grams of raw material, one hour of labor type $A$, two hours of labor type $B$. Because of the higher price, the market demand for $U$ is limited to forty units per week. $V$ is the lower priced version of the product that sells for 210€ per unit with unlimited demand and requires 9 grams of raw material, 1 hour of labor type $A$ and 1 hour of labor type $B$. This data is summarized in the following table:
# 
# | Version | Raw Material <br> required | Labor A <br> required | Labor B <br> required | Market <br> Demand | Price |
# | :-: | :-: | :-: | :-: | :-: | :-: | 
# | U | 10 g | 1 hr | 2 hr | $\leq$ 40 units | 270€ |
# | V |  9 g | 1 hr | 1 hr | unlimited | 210€ |
# 
# Weekly production at the company is limited by the availability of labor and the inventory of raw material. The raw material must be ordered in advance and has a short shelf life. Any raw material left over at the end of the week is discarded. The following table details the cost and availability of raw material and labor.
# 
# | Resource | Amount <br> Available | Cost | 
# | :-: | :-: | :-: |
# | Raw Material | ? | 10€ / g |
# | Labor A | 80 hours | 50€ / hour |
# | Labor B | 100 hours | 40€ / hour | 
# 
# The company wishes to maximize gross profits. How much raw material should be ordered in advance for each week? How many units of $U$ and $V$ should the company produce each week? 
# 
# ### Mathematical Model
# 
# A starting point for the development of a mathematical model is to create a list of relevant decision variables. Some of decision variables may prove redundant later, tut for now we seek to identify quantities that may be useful in expressing the problem objectives and constraints. A candidate set of decision variable is listed in the following table with a symbol, description, and any lower and upper bounds that are known from the problem data.
# 
# | Decision Variable | Description | lower bound | upper bound |
# | :-: | :--- | :-: | :-: |
# | $x_M$ | amount of raw material used | 0 | - |
# | $x_A$ | amount of Labor A used | 0 | 80 |
# | $x_B$ | amount of Labor B used | 0 | 100 |
# | $y_U$ | number of $U$ units to produce | 0 | 40 |
# | $y_V$ | number of $V$ units to product | 0 | - |
# | revenue | amount of revenue generated by sales | - | - |
# | expense | cost of raw materials and labor | - | - |
# | profit | difference between revenue and expense | - | - |
# 
# The problem objective is to maximize profit
# 
# $$\max\ \text{profit}$$
# 
# where profit is defined as the difference between revenue and expense. 
# 
# $$
# \begin{aligned}
# \text{profit} & = \text{revenue} - \text{expense} \\
# \end{aligned}
# $$
# 
# Revenue and profit are linear expressions that can be written in terms of the decision variables.
# 
# $$
# \begin{aligned}
# \text{revenue} & = 270 y_U + 210 y_V \\
# \text{expense} & = 10 x_M + 50 x_A + 40 x_B  \\
# \end{aligned}
# $$
# 
# The quantity of products produced is limited by the available resources. For each resource there is a corresponding linear constraint that limits production.
# 
# $$
# \begin{aligned}
# 10 y_U + 9 y_V  & \leq x_M & & \text{raw material}\\
# 2 y_U + 1 y_V & \leq x_A & &\text{labor A} \\
# 1 y_U + 1 y_V & \leq x_B & & \text{labor B}\\
# \end{aligned}
# $$
# 
# This completes the mathematical description of this example of a production planning problem. The next is create a Pyomo model t

# ## Version 1. Getting started with Pyomo
# 
# This first version of a Pyomo model uses components from the Pyomo library to represent the decision variables, expressions, objectives, and constraints appearing in the mathematical model for this production planning problem. This is a direct translation of the mathematical model to Pyomo. Subsequent versions of this model will demonstrate the use of indexed parameters, variables and constraints that are essential for larger scale applications.

# **Step 0.Installing Pyomo and solvers.**
# 
# Before going further, the first step is to install the Pyomo library and any solvers that may be used to compute numerical solutions. The following cell downloads a Python module that will check for (and if necessary install) Pyomo and a linear solver on Google Colab and most laptops.

# In[24]:


import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# **Step 1. Import Pyomo.** 
# 
# The first step in creating a Pyomo model is to import the needed components from the Pyomo library into the Python environment. Importing  `pyomo.environ` will provides the most commonly used components for Pyomo model building. 
# 
# This collection of notebooks uses a consistent convention of assigning a `pyo` prefix to objects imported from `pyomo.environ`.

# In[36]:


import pyomo.environ as pyo


# **Step 2. Create a `ConcreteModel` object.** 
# 
# A Pyomo model is stored in a Python workspace with a standard Python variable name. The model object can be created with either `pyo.ConcreteModel()` or `pyo.AbstractModel()`. `pyo.ConcreteModel` is used when the problem data is known at the time the model is created, which is the case here. `pyo,AbstractModel` is useful for situations where the model is created before the model data is known. 
# 
# The following cell creates a ConcreteModel object and stores it in a Python variable named `model`. The name can be any valid Python variable, but keep in mind the name will be a prefix for every variable and constraint. The `.display()` method is a convenient means of displaying the current contents of a Pyomo model.

# In[40]:


model = pyo.ConcreteModel("Production Planning: Version 1")
model.display()


# **Step 3. Create decision variables and add them to the model.**
# 
# Decision variables are created with the Pyomo `Var()` class. Variables are assigned as attributes to the model object using the Python 'dot' convention. 
# 
# `Var()` accepts optional keyword arguments. The optional `bounds` keyword specifies a tuple containing lower and upper bounds. A good modeling practice is to specify any known and fixed bounds on the decision variables as they are created. If one of the bounds is unknown, use `None` as a placeholder.
# 
# In addition, the `initialize` keyword is used below to specify initial values for the decision variables. This isn't normally required, but is useful for this tutorial example where we want to display the model as it is being built.
# 
# The optional `domain` keyword specifies the type of decision variable. By default, the domain is all real numbers including negative and positive values. 

# In[41]:


model.x_M = pyo.Var(bounds=(0, None), initialize=0)
model.x_A = pyo.Var(bounds=(0, 80), initialize=0)
model.x_B = pyo.Var(bounds=(0, 100), initialize=0)

model.y_U = pyo.Var(bounds=(0, 40), initialize=0)
model.y_V = pyo.Var(bounds=(0, None), initialize=0)

model.display()


# **Step 4. Create expressions.**
# 
# The difference between revenue and expense is equal to the gross profit. The following cell creates linear expressions for revenue and expense which are then assigned the expressions to attributes called `model.revenue` and `model.expense`. We can print the expressions to verify correctness.

# In[42]:


model.revenue = 270 * model.y_U + 210 * model.y_V
model.expense = 10 * model.x_M + 50 * model.x_A + 40 * model.x_B

print(model.revenue)
print(model.expense)


# **Step 5. Add an objective.**
# 
# The problem objective is to maximize gross profit, where gross profit is the difference between revenue and expense.
# 
# The Pyomo class `Objective` creates the objective for a Pyomo model. The required keyword argument `expr` specifies the expression to be optimized. The optional keyword argument `sense` specifies whether it is a minimization or maximization problem. For clarity, it is good practice is to always include the `sense` keyword argument.
# 
#     model.profit = pyo.Objective(expr = model.revenue - model.expense, sense = pyo.maximize)
#     
# An alternative way to create an objective is available in more recent releases of Pyomo. This approach uses a Python decorators to 'tag' function that returns an Pyomo expression as the objective function. 
# 
#     @model.Objective(sense = pyo.maximize)
#     def profit(model):
#         return model.revenue - model.expense
# 
# Decorators are a feature of Python that change the behavior of a function without altering the function itself. The Pyomo library includes decorators to declare objectives, expression, constraints, and other objects. The use of decorators improves the readability and maintainability of more complex models. For now, one can think of decorators as a means of tagging a function with for the purpose of create model objects.
# 
# The name of the function will be used to create a model attribute. When passed as the first argument to the function, the model object will allow access to other model attributes, such as expressions or decision variables. Arguments can be passed to the decorator to specify the purpose of the function. For example, the `sense` keyword is used to set a maximization objective.
# 

# In[43]:


@model.Objective(sense=pyo.maximize)
def profit(model):
    return model.revenue - model.expense

model.display()


# **Step 5. Add constraints.**
# 
# Constraints are logical relationships between expressions that define the range of feasible solutions in an optimization problem.. The logical relationships can be `==`, `<=`, or `>=`. 
# 
# `pyo.Constraint()` is a Python library component that provides a convenient way to define and enforce constraints on scalar values.  When using`pyo.Constraint()`, a constraint consisting of a logical relationship between expressions is passed as a keyword argument `expr`. For the three equality constraints in this application, this appears as  
# 
#     model.raw_materials = pyo.Constraint(expr = 10 * model.y_U + 9 * model.y_V <= model.x_M)
#     model.labor_A = pyo.Constraint(expr = 2 * model.y_U + 1 * model.y_V <= model.x_A)
#     model.labor_B = pyo.Constraint(expr = 1 * model.y_U + 1 * model.y_V <= model.x_B)
#     
# Decorators, however, allow for a more readable and maintainable way to express complex an indexed constraints in a model. In this case, the constraints are
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
# 

# In[35]:


@model.Constraint()
def raw_materials(model):
    return 10 * model.y_U + 9 * model.y_V <= model.x_M

@model.Constraint()
def labor_A(model):
    return 2 * model.y_U + 1 * model.y_V <= model.x_A

@model.Constraint()
def labor_B(model):
    return 1 * model.y_U + 1 * model.y_V <= model.x_B

model.display()


# **Step 6. Solve the model.**

# In[34]:


solver = pyo.SolverFactory("cbc")
solver.solve(model)

model.display()


# **Display the solution.**
# 

# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 12))
ax = plt.subplot(211, xlabel="qty y1", ylabel="qty y2", 
                 aspect=1, xlim=(0, 80), ylim=(0, 80))

u = np.linspace(0, 80)

ax.plot(u, (80 - u), lw=2, label="labor A")
ax.plot(u, (100 - 2*u), lw=2, label="labor B")
ax.plot([40]*2, ax.get_ylim(), lw=2, label="demand")

ax.fill_between(u, (80 - u), 100, alpha=0.2)
ax.fill_between(u, (100 - 2*u), 100, alpha=0.2)
ax.fill_between([40, ax.get_xlim()[1]], [0]*2, [100]*2, alpha=0.2)
ax.fill_between([ax.get_xlim()[0], 20], [0]*2, [100]*2, alpha=0.2)

ax.legend(loc="upper right")

for profit in [600*k for k in range(20)]:
    ax.plot(u, (profit - 40*u)/30, 'r:', alpha=1, lw=0.5)

ax.plot(m.y1.value, m.y2.value, 'r.', ms=20)


# ## Organizing Data
# 

# In[186]:


import pandas as pd

products = {
    "U": {"price": 270, "demand": 40},
    "V": {"price": 210, "demand": None},
}

resources = {
    "M": {"price": 10, "available": None},
    "labor A": {"price": 50, "available": 100},
    "labor B": {"price": 40, "available":  80},
}

uses = {
    "U": {"M": 10, "labor A": 2, "labor B": 1},
    "V": {"M":  9, "labor A": 1, "labor B": 1},
}


# In[190]:


model = pyo.ConcreteModel("Production Planning: Version 2")


model.PRODUCTS = pyo.Set(initialize=products.keys())
model.RESOURCES = pyo.Set(initialize=resources.keys())

model.x = pyo.Var(model.RESOURCES, bounds=(0, None), initialize=0)
for r in model.RESOURCES:
    model.x.ub = resources[r]["available"]
    
model.display()


# In[ ]:





# In[ ]:




