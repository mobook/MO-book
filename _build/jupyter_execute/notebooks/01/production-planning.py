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
# # Pyomo Tutorial Example: Production Planning
# 
# Pyomo is an algebraic modeling language embedded within Python for creating mathematical optimization applications. Pyomo can create objects representing decision variables, expressions, objective functions, and constraints. Much of the utility of Pyomo comes from methods to transform and solve models with a choice of open-source and commercial solvers. 
# 
# The purpose of this notebook is to introduce the basic elements of Pyomo with a case study of a small production planning problem. A more complete version of the problem is revisited in Chapter 10 to demonstrate stochastic and robust optimization. This notebook introduces components from the Pyomo library common to most applications:
# 
# * [Variables](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Variables.html)
# * [Objectives](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Objectives.html)
# * [Constraints](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Constraints.html)
# * [Expressions](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html)
# <<<<<<< local
# * [SolverFactory]()
# =======
# * [Parameters](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Parameters.html)
# * [Sets](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Sets.html)
# >>>>>>> remote
# 
# The notebook begins with a statement of production planning problem and presentation of a mathematical model. The mathematical model is then translated into series of Pyomo models of increasing abstraction.
# 
# The first version of the model is a direct translation of the mathematics into corresponding Pyomo components using `Var()`, `Objective()`, `Constraint` and `SolverFactory`. No attempt is made at generalizing the model, and parameter values are used directly in expressions. 
# 
# Version 2 refactors the Pyomo model by using Python data structures to represent problem data, and employing `Set()` to create indices for decision variables and constraints. This provides a model that can be reused for 
# 
# Version 3 of the model extends 
# 
# This generates a satisfactory solution and would be enough for examples involving a small number of decision variables and constraints. This version of the model provides a tutorial introduction to the use of variables, expressions, objectives, and constraints.
# 
# Version 2 of the model introduces the use of sets, parameters, indexed variables, and indexed constraints. These additional components of the Pyomo library are essential to building scalable and maintainable models for more complex applications. 
# 
# One feature of these notebooks is the use of Python decorators to designate Pyomo objectives, constraints, and other model components. This is a relatively new feature available in recent versions of Pyomo. Decorators are a less commonly used feature of Python that may be unfamiliar to new users, but are well worth the effort of learning for a remarkable gain in the readability and maintenance of Pyomo models.
# 
# So let's get started.

# <span style="color:red">**<<<<<<< local**</span>

# ## A Production Planning Problem
# 
# A company produces two versions of a product. Each version is made from the same raw material that costs 10€ per gram, and each version requires two different types of specialized labor to finish. 
# 
# $U$ is the higher priced version of the product. $U$ sells for  270€ per unit and requires 10 grams of raw material, one hour of labor type $A$, two hours of labor type $B$. The market demand for $U$ is limited to forty units per week. $V$ is the lower priced version of the product with unlimited demand that sells for 210€ per unit and requires 9 grams of raw material, 1 hour of labor type $A$ and 1 hour of labor type $B$. This data is summarized in the following table:
# 
# | Version | Raw Material <br> required | Labor A <br> required | Labor B <br> required | Market <br> Demand | Price |
# | :-: | :-: | :-: | :-: | :-: | :-: | 
# | U | 10 g | 1 hr | 2 hr | $\leq$ 40 units | 270€ |
# | V |  9 g | 1 hr | 1 hr | unlimited | 210€ |
# 
# Weekly production at the company is limited by the availability of labor and by the inventory of raw material. The raw material has a shelf life of one week and must be ordered in advance. Any raw material left over at the end of the week is discarded. The following table details the cost and availability of raw material and labor.
# 
# | Resource | Amount <br> Available | Cost | 
# | :-: | :-: | :-: |
# | Raw Material | ? | 10€ / g |
# | Labor A | 80 hours | 50€ / hour |
# | Labor B | 100 hours | 40€ / hour | 
# 
# The company wishes to maximize gross profits. 
# 
# 1. How much raw material should be ordered in advance for each week? 
# 2. How many units of $U$ and $V$ should the company produce each week? 

# ## Mathematical Model
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

# <span style="color:red">**=======**</span>

# ## A Production Planning Problem

# ### Problem Statement
# 
# A small company has developed two versions of a new product. Each version of the product is made from the same raw material that costs 10€ per gram, and requires two different types of specialized labor to produce. 
# 
# $U$ is the higher priced version of the product. $U$ sells for  270€ per unit and requires 10 grams of raw material, one hour of labor type $A$, two hours of labor type $B$. Because of the higher price, the market demand for $U$ is limited to forty units per week. $V$ is the lower priced version of the product that sells for 210€ per unit with unlimited demand and requires 9 grams of raw material, 1 hour of labor type $A$ and 1 hour of labor type $B$. This data is summarized in the following table:
# 
# | Version | Raw material <br> required | Labor A <br> required | Labor B <br> required | Market <br> demand | Price |
# | :-: | :-: | :-: | :-: | :-: | :-: | 
# | U | 10 g | 1 hr | 2 hr | $\leq$ 40 units | 270€ |
# | V |  9 g | 1 hr | 1 hr | unlimited | 210€ |
# 
# Weekly production at the company is limited by the availability of labor and the inventory of raw material. The raw material must be ordered in advance and has a short shelf life. Any raw material left over at the end of the week is discarded. The following table details the cost and availability of raw material and labor.
# 
# | Resource | Amount <br> available | Cost | 
# | :-: | :-: | :-: |
# | Raw Material | ? | 10€ / g |
# | Labor A | 80 hours | 50€ / hour |
# | Labor B | 100 hours | 40€ / hour | 
# 
# The company wishes to maximize gross profits. How much raw material should be ordered in advance for each week? How many units of $U$ and $V$ should the company produce each week? 

# ### Mathematical Model
# 
# The above is an example of an optimization problem. Formulating such a problem in a mathematical fashion always involves three crucial elements: **decision variables, constraints, and an objective function**, which we will now introduce.
# 
# The starting point for the development of a mathematical model is to create a list of relevant decision variables. Some of the decision variables may prove redundant later, tut for now we seek to identify quantities that may be useful in expressing the problem objectives and constraints. A candidate set of decision variable is listed in the following table with a symbol, description, and any lower and upper bounds that are known from the problem data.
# 
# | Decision variable | Description | Lower bound | Upper bound |
# | :-: | :--- | :-: | :-: |
# | $x_M$ | amount of raw material used | 0 | - |
# | $x_A$ | amount of Labor A used | 0 | 80 |
# | $x_B$ | amount of Labor B used | 0 | 100 |
# | $y_U$ | number of $U$ units to produce | 0 | 40 |
# | $y_V$ | number of $V$ units to product | 0 | - |
# 
# Using the decision variables above, we are able to formulate the **objective function** to be maximized -- the profit,
# 
# $$\max\ \text{profit}$$
# 
# which is equal to the difference between revenue and the costs:which equals the difference between revenue and the costs:
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
# The decision variables $y_U, y_V, x_M ,x_A ,x_B$ cannot take any value, but they need to specify specific conditions. maximize the profit formula we cannot, however, just pick any values. Instead, they have to satisfy certain **constraints**, expressed as equalities or inequalities involving the decision variables. For each resource there is a corresponding linear constraint that limits production, namely:
# 
# $$
# \begin{aligned}
# 10 y_U + 9 y_V  & \leq x_M & & \text{(raw material)}\\
# 2 y_U + 1 y_V & \leq x_A & &\text{(labor A)} \\
# 1 y_U + 1 y_V & \leq x_B & & \text{(labor B)}\\
# \end{aligned}
# $$
# 
# We are now ready to formulate the full mathematical optimization problem in the following canonical way: first we state the objective function to be maximized (or minimized), then we list all the constraints, and lastly we report all the decision variables and their bounds:
# 
# $$
# \begin{align}
# \max \quad & 270 y_U + 210 y_V - 10 x_M - 50 x_A - 40 x_B \\
# \text{such that = s.t.} \quad & 10 y_U + 9 y_V  \leq x_M \nonumber \\
#  & 2 y_U + 1 y_V \leq x_A \nonumber \\
#  & 1 y_U + 1 y_V \leq x_B \nonumber \\
#  & 0 \leq x_M \nonumber \\
#  & 0 \leq x_A \leq 80 \nonumber \\
#  & 0 \leq x_B \leq 100 \nonumber \\
#  & 0 \leq y_U \leq 40 \nonumber \\
#  & 0 \leq y_V.\nonumber 
# \end{align}
# $$
# 
# This completes the mathematical description of this example of a production planning problem. 
# 
# In many textbooks it is customary to write the decision variables also under the $\min$ symbol so as to clearly distinguish between the variables and any other parameters (which might also be expressed as symbols or letters) in the problem formulation. Throughout this book, however, we stick to the convention that the decision variables are only those for which explicitly state their domains as part of the problem formulation.
# 
# An **optimal solution** of the problem is any vector of decision variables that meets the constraints and achieves the maximum/minimum objective.
# 
# However, even for a simple problem like this one, it is not immediately clear what the optimal solution is. This is exactly where mathematical optimization algorithms come into play -- they are generic procedures that can find the optimal solutions of problems as long as these problems can be formulated in a standardized fashion as above. For a practitioner, mathematical optimization very often boils down to formulating the problem as a model above, and then passing it over to one of the open-source or commercial software packages that can solve such a model, regardless of what was the original *story* behind the model. In order to do so, we need an interface of communication between the models and the algorithms. In this book, we opt for a Python-based interface which is the ```Pyomo``` modelling package.
# 
# The next step is thus creating the corresponding Pyomo model.

# <span style="color:red">**>>>>>>> remote**</span>

# ## Version 1: Scalar Variables and Constraints
# 
# This first version of a Pyomo model uses components from the Pyomo library to represent the decision variables, expressions, objectives, and constraints as they appear in the mathematical model for this production planning problem. This is a direct translation of the mathematical model to Pyomo using the basic elements of Pyomo.

# <<<<<<< local
# ### Step 0.Install Pyomo and solvers
# =======
# **Step 0. Installing Pyomo and solvers.**
# >>>>>>> remote
# 
# Before going further, the first step is to install the Pyomo library and any solvers that may be used to compute numerical solutions. The following cell downloads a Python module that will check for (and if necessary install) Pyomo and a linear solver on Google Colab and most laptops.

# In[ ]:


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

# In[ ]:


import pyomo.environ as pyo


# ### Step 2. Create a `ConcreteModel` object
# 
# A Pyomo model is stored in a Python workspace with a standard Python variable name. The model object can be created with either `pyo.ConcreteModel()` or `pyo.AbstractModel()`. `pyo.ConcreteModel` is used when the problem data is known at the time the model is created, which is the case here. `pyo,AbstractModel` is useful for situations where the model is created before the model data is known. 
# 
# The following cell creates a ConcreteModel object and stores it in a Python variable named `model`. The name can be any valid Python variable, but keep in mind the name will be a prefix for every variable and constraint. 

# In[ ]:


model = pyo.ConcreteModel("Production Planning: Version 1")


# The `.display()` method is a convenient means of displaying the current contents of a Pyomo model.

# In[23]:


model.display()


# <<<<<<< local
# ### Step 3. Create decision variables
# =======
# **Step 3. Create and add decision variables.**
# >>>>>>> remote
# 
# Decision variables are created with the Pyomo `Var()` class. Variables are assigned to attributes of the model object using the Python 'dot' notation. `Var()` accepts optional keyword arguments. 
# 
# * The optional `bounds` keyword specifies a tuple containing lower and upper bounds. A good modeling practice is to specify any known and fixed bounds on the decision variables as they are created. If one of the bounds is unknown, use `None` as a placeholder.
# 
# <<<<<<< local
# * The `initialize` keyword specifies initial values for the decision variables. This isn't normally required, but is useful in this tutorial example where we want to display the model as it is being built.
# 
# * The optional `domain` keyword specifies the type of decision variable. By default, the domain is all real numbers including negative and positive values. 
# =======
# In addition, the `initialize` keyword is used below to specify initial values for the decision variables. This is not normally required, but is useful for this tutorial example where we want to display the model as it is being built.
# 
# The optional `domain` keyword specifies the type of decision variable. By default, the domain is all real numbers, hence including both negative and positive values. 
# >>>>>>> remote

# In[ ]:


model.x_M = pyo.Var(bounds=(0, None), initialize=0)
model.x_A = pyo.Var(bounds=(0, 80), initialize=0)
model.x_B = pyo.Var(bounds=(0, 100), initialize=0)

model.y_U = pyo.Var(bounds=(0, 40), initialize=0)
model.y_V = pyo.Var(bounds=(0, None), initialize=0)

model.display()


# ### Step 4. Create expressions
# 
# The difference between revenue and expense is equal to the gross profit. The following cell creates linear expressions for revenue and expense which are then assigned the expressions to attributes called `model.revenue` and `model.expense`. We can print the expressions to verify correctness.

# In[ ]:


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

# In[ ]:


@model.Objective(sense=pyo.maximize)
def profit(model):
    return model.revenue - model.expense

model.display()


# **Step 6. Add the constraints.**
# 
# Constraints are logical relationships between expressions that define the range of feasible solutions in an optimization problem. The logical relationships can be `==`, `<=`, or `>=`. 
# 
# `pyo.Constraint()` is a Pyomo library component to creating scalar constraints. A constraint consists of a logical relationship between expressions is passed as a keyword argument `expr` to `pyo.Constraint()`. For this application, the constraints are expressed as  
# 
#     model.raw_materials = pyo.Constraint(expr = 10 * model.y_U + 9 * model.y_V <= model.x_M)
#     model.labor_A = pyo.Constraint(expr = 2 * model.y_U + 1 * model.y_V <= model.x_A)
#     model.labor_B = pyo.Constraint(expr = 1 * model.y_U + 1 * model.y_V <= model.x_B)
#     
# Alternatively, the decorator syntax for constraints provides a readable and maintainable means to express more complex constraints. For the present example, the constraints would be written
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

# In[ ]:


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


# **Step 7. Solve the model.**
# 
# With the model now fully specified, the next step is to compute a solution. This is done by creating a solver object using `SolverFactory`, then applying the solver to the model, as shown in the following cell.
# 
# EXPAND DISCUSSION REGARDING SOLVERS.

# In[ ]:


solver = pyo.SolverFactory("cbc")
solver.solve(model)

model.pprint()


# **Step 8. Reporting the solution.**
# 
# WRITE THIS STEP.
# 
# For an overview of other ways to report and visualize the solutions, see also the appendix of [this notebook](../04/gasoline-distribution.ipynb).

# ## Version 2: Creating a Data-Driven Model
# 

# In[9]:


import pandas as pd

product_data = {
    "U": {"price": 270, "demand": 40},
    "V": {"price": 210, "demand": None},
}

display(pd.DataFrame(product_data))

resource_data = {
    "M": {"price": 10, "available": None},
    "labor A": {"price": 50, "available": 100},
    "labor B": {"price": 40, "available":  80},
}

display(pd.DataFrame(resource_data))

process_data = {
    "U": {"M": 10, "labor A": 2, "labor B": 1},
    "V": {"M":  9, "labor A": 1, "labor B": 1},
}

display(pd.DataFrame(process_data))


# **Enhancement 1: Create sets of products and resources**

# In[10]:


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


# To learn how to write consistent, readable, and maintainable Pyomo code for modeling and optimizationconsistent, see the [Pyomo style guide](../../pyomo_style_guide.md)
