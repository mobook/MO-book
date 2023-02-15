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

# # A Production Planning Problem

# ## Problem Statement
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
# In many textbooks it is customary to list the decision variables under the $\min$ symbol so as to clearly distinguish between the variables and any other parameters (which might also be expressed as symbols or letters) in the problem formulation. Throughout this book, however, we stick to the convention that the decision variables are only those for which explicitly state their domains as part of the problem formulation.
# 
# An **optimal solution** of the problem is any vector of decision variables that meets the constraints and achieves the maximum/minimum objective.
# 
# However, even for a simple problem like this one, it is not immediately clear what the optimal solution is. This is exactly where mathematical optimization algorithms come into play - they are generic procedures that can find the optimal solutions of problems as long as these problems can be formulated in a standardized fashion as above. 
# 
# For a practitioner, mathematical optimization often boils down to formulating the problem as a model above, and then passing it over to one of the open-source or commercial software packages that can solve such a model regardless of what was the original *story* behind the model. To do so, we need an interface of communication between the models and the algorithms. In this book, we opt for a Python-based interface which is the ```Pyomo``` modeling package.
# 
# The next step is to create the corresponding Pyomo model.

# In[ ]:




