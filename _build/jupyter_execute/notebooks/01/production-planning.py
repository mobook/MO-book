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
# * [Sets](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Sets.html)
# * [Parameters](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Parameters.html)
# * [Variables](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Variables.html)
# * [Objectives](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Objectives.html)
# * [Constraints](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Constraints.html)
# * [Expressions](https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html)
# * [SolverFactory]()
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
# These notebooks use Python decorators to designate Pyomo objectives, constraints, and other model components. This is a new feature available in recent versions of Pyomo.  Decorators may be unfamiliar to new users of Python, but are  worth the effort of learning for a remarkable gain in the readability of Pyomo models.
# 
# So let's get started.

# ## A Production Planning Problem

# ### Problem Statement
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
# The problem statement above describes an optimization problem. Reformulating the problem statement as a mathematical model involves a few crucial elements:
# 
# * **decision variables**,
# * **expressions**,
# * **objective function**,
# * **constraints**.
# 
# The usual starting point for a developing a mathematical model is to create a list of relevant decision variables. Decision variables are quantities the problem statement that that can be changed to achieve a desired result.  of the decision variables introduced at this stage may prove unnecessarybut for now we seek a list of variables that will be useful in expressing problem objective and constraints. 
# 
# For the problem given above, a any lower and upper bounds that are known from the problem data. candidate set of decision variable is listed in the following table with a symbol, description, and later, Some
# 
# | Decision Variable | Description | lower bound | upper bound |
# | :-: | :--- | :-: | :-: |
# | $x_M$ | amount of raw material used | 0 | - |
# | $x_A$ | amount of Labor A used | 0 | 80 |
# | $x_B$ | amount of Labor B used | 0 | 100 |
# | $y_U$ | number of $U$ units to produce | 0 | 40 |
# | $y_V$ | number of $V$ units to product | 0 | - |
# 
# The next step is to formulate an **objective function** describing the metric that will be used to quantify a solution to the problem. In this, that quantify is profit which is to be maximized. 
# 
# $$\max\ \text{profit}$$
# 
# where profit is equal to the difference between revenue and cost of operations: 
# 
# $$
# \begin{aligned}
# \text{profit} & = \text{revenue} - \text{cost} \\
# \end{aligned}
# $$
# 
# Revenue and cost are linear **expressions** that can be written in terms of the decision variables.
# 
# $$
# \begin{aligned}
# \text{revenue} & = 270 y_U + 210 y_V \\
# \text{cost} & = 10 x_M + 50 x_A + 40 x_B  \\
# \end{aligned}
# $$
# 
# As shown here, an expression is a algebraic combination of variables that can be referred to by a name. Expressions are useful when the same combination of variables appears in multiple places in a model, or when it is desirable to break up longer expressions into smaller units. Here, for example, we create expressions for revenue and cost that simplify the objective function.
# 
# The decision variables $y_U, y_V, x_M, x_A, x_B$ need to satisfy the specific conditions given in the problem statement. **Constraints** are mathematical relationships among decision variables or expressions. In this problem, for exaample, for each resource there is a corresponding linear constraint that limits production:
# 
# $$
# \begin{aligned}
# 10 y_U + 9 y_V  & \leq x_M & & \text{raw material}\\
# 2 y_U + 1 y_V & \leq x_A & &\text{labor A} \\
# 1 y_U + 1 y_V & \leq x_B & & \text{labor B}\\
# \end{aligned}
# $$
# 
# We are now ready to formulate the full mathematical optimization problem in the following canonical way: First we state the objective function to be maximized (or minimized), then we list all the constraints, and lastly we list all the decision variables and their bounds:
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
# However, even for a simple problem like this one, it is not immediately clear what the optimal solution is. This is exactly where mathematical optimization algorithms come into play - they are generic procedures that can find the optimal solutions of problems as long as these problems can be formulated in a standardized fashion as above. For a practitioner, mathematical optimization often boils down to formulating the problem as a model above, and then passing it over to one of the open-source or commercial software packages that can solve such a model regardless of what was the original *story* behind the model. In order to do so, we need an interface of communication between the models and the algorithms. In this book, we opt for a Python-based interface which is the ```Pyomo``` modeling package.
# 
# The next step is thus creating the corresponding Pyomo model.

# ## Pyomo Model Version 1: Scalar Variables and Constraints
# 
# This first version of a Pyomo model uses components from the Pyomo library to represent decision variables, expressions, objectives, and constraints as they appear in the mathematical model for this production planning problem. This is a direct translation of the mathematical model to Pyomo using the basic elements of Pyomo. 

# ### Preliminary Step: Install Pyomo and solvers
# 
# Before going further it is necessary to install the Pyomo library and any solvers that will be used to compute numerical solutions. The following cell downloads a Python module that will check for (and if necessary install) Pyomo and a linear solver. This cell will work on Google Colab and most laptops. The installation needs to be one-time for a laptop, and done for each new Google Colab session.

# In[1]:


import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ### Step 1. Import Pyomo
# 
# The first coding step in creating a Pyomo model is to import the needed components from the Pyomo library into the Python environment. Importing  `pyomo.environ` will provides the most commonly used components for Pyomo model building. 
# 
# This collection of notebooks uses a consistent convention of assigning a `pyo` prefix to objects imported from `pyomo.environ`.

# In[2]:


import pyomo.environ as pyo


# ### Step 2. Create a `ConcreteModel` object
# 
# A Pyomo model is stored in a Python workspace with a standard Python variable name. The model object can be created with either `pyo.ConcreteModel()` or `pyo.AbstractModel()`. `pyo.ConcreteModel` is used when the problem data is known at the time the model is created, which is the case here. 
# 
# The following cell creates an instance of ConcreteModel object and stores it in a Python variable named `model`. The name can be any valid Python variable, but keep it short since the name will be a prefix for every Pyomo variable and constraint. An optional string is used to label subsequent model display.

# In[3]:


# create model
model = pyo.ConcreteModel("Production Planning: Version 1")


# The `.display()` method is a convenient means of displaying the current contents of a Pyomo model.

# In[4]:


#display model
model.display()


# ### Step 3. Create decision variables
# 
# Decision variables are created with the Pyomo `Var()` class and assigned to a model instance using the Python 'dot' notation. 
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


# ### Step 4. Create expressions
# 
# The next cell creates linear expressions for revenue and cost and assigns them to the model as `model.revenue` and `model.cost`, respectively. These expressions will be used to create the profit objective. The expressions can be printed to verify correctness.

# In[6]:


# create expressions
model.revenue = 270 * model.y_U + 210 * model.y_V
model.cost = 10 * model.x_M + 50 * model.x_A + 40 * model.x_B

# expressions can be printed
print(model.revenue)
print(model.cost)


# ### Step 5. Create objective
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


# ### Step 6. Add constraints
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


# ### Step 7. Solve the model
# 
# With the model now fully specified, the next step is to compute a solution. A solver object is created with `SolverFactory`, then applied to the model as shown in the following cell. Here we have chosen to use the open source [COIN-OR Cbc](https://github.com/coin-or/Cbc)  ("COIN-OR branch and cut") solver for mixed integer linear programming. There are suitable solver, such as the open source GNU Linear Programming Kit [GLPK](https://en.wikibooks.org/wiki/GLPK) and commercial solvers such as CPLEX, Gurobi, and Mosek.

# In[ ]:


solver = pyo.SolverFactory("cbc")
solver.solve(model)

model.pprint()


# ### Step 8. Reporting the solution
# 
# WRITE THIS STEP.
# 
# For an overview of other ways to report and visualize the solutions, see also the appendix of [this notebook](../04/gasoline-distribution.ipynb).

# ## Data Representation
# 
# Choosing an organized and complete representation of the problem data is a productive starting point for creating Pyomo models. In this case.  the data consists of (1) numbers describing the price and demand for products, (2) numbers for the cost and availability of resources needed to produce the products, and (3) numbers describing the amount of resources needed to produce each unit of every product. This suggests three tables to represent the problem data.
# 
# Here we use nested dictionaries as containers for the problem data, and use Pandas DataFrames to display the data. For the product data, the dictionary keys are the names of each product. For each product there is an associated dictionary containing the price and market demand. `None` is used as a placeholder if there is no limit on market demand.

# In[6]:


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

# In[7]:


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




