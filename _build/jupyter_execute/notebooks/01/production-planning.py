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

# # A first production planning problem

# ## Problem statement
# 
# A company produces two versions of a product. Each version is made from the same raw material that costs 10€ per gram, and each version requires two different types of specialized labor to finish. $U$ is the higher priced version of the product. $U$ sells for 270€ per unit and requires 10 grams of raw material, one hour of labor type $A$, two hours of labor type $B$. Due to the higher price, the market demand for $U$ is limited to 40 units per week. $V$ is the lower priced version of the product with unlimited demand that sells for 210€ per unit and requires 9 grams of raw material, 1 hour of labor type $A$ and 1 hour of labor type $B$. This data is summarized in the following table:
# 
# <div align="center">
# 
# | Version | Raw Material <br> required | Labor A <br> required | Labor B <br> required | Market <br> Demand | Price |
# | :-: | :-: | :-: | :-: | :-: | :-: | 
# | U | 10 g | 1 hr | 2 hr | $\leq$ 40 units | 270€ |
# | V |  9 g | 1 hr | 1 hr | unlimited | 210€ |
# 
# </div>
# 
# Weekly production at the company is limited by the availability of labor and the inventory of raw materials. Raw material has a shelf life of one week and must be ordered in advance. Any raw material left over at the end of the week is discarded. The following table details the cost and availability of raw material and labor.
# 
# <div align="center">
# 
# | Resource | Amount <br> Available | Cost | 
# | :-: | :-: | :-: |
# | Raw Material | unlimited | 10€ / g |
# | Labor A | 80 hours/week | 50€ / hour |
# | Labor B | 100 hours/week | 40€ / hour | 
# 
# </div>
# 
# The company wants to maximize its gross profits. 
# 
# 1. How much raw material should be ordered in advance for each week? 
# 2. How many units of $U$ and $V$ should the company produce each week? 

# ## Mathematical model
# 
# The problem statement above describes an optimization problem. Reformulating the problem statement as a mathematical model involves a few crucial elements:
# 
# * **decision variables**,
# * **expressions**,
# * **objective function**,
# * **constraints**.
# 
# The starting point is developing a mathematical model is to list decision variables relevant to the problem at hand. **Decision variables** are quantities that can be modified to achieve a desired outcome.  While some decision variables introduced at this stage may prove redundant later, the goal at this point is to create a comprehensive list of variables that will be useful in expressing the problem's objective and constraints. 
# 
# For this problem statement, listed below are decision variables with symbols, descriptions, and any lower and upper bounds that are known from the problem data.
# 
# <div align="center">
# 
# | Decision Variable | Description | lower bound | upper bound |
# | :-: | :--- | :-: | :-: |
# | $x_M$ | amount of raw material used | 0 | - |
# | $x_A$ | amount of Labor A used | 0 | 80 |
# | $x_B$ | amount of Labor B used | 0 | 100 |
# | $y_U$ | number of $U$ units to produce | 0 | 40 |
# | $y_V$ | number of $V$ units to product | 0 | - |
# 
# </div>
# 
# The next step is to formulate an **objective function** describing that describes how we will measure the value of candidate solutions to the problem. In this case, the value of the solution is measured by profit which is to be maximized. 
# 
# $$\max\ \text{profit}$$
# 
# Profit, in turn, is equal to the difference between revenue and cost of operations: 
# 
# $$
# \begin{aligned}
#     \text{profit} & = \text{revenue} - \text{cost} \\
# \end{aligned}
# $$
# 
# Revenue and cost are linear **expressions** that can be written in terms of the decision variables.
# 
# $$
# \begin{aligned}
#     \text{revenue} & = 270 y_U + 210 y_V \\
#     \text{cost} & = 10 x_M + 50 x_A + 40 x_B  \\
# \end{aligned}
# $$
# 
# As shown here, an expression is an algebraic combination of variables that can be referred to by name. Expressions are useful when the same combination of variables appears in multiple places in a model, or when it is desirable to break up longer expressions into smaller units. Here we have created expressions for revenue and cost that simplify the objective function.
# 
# The decision variables $y_U, y_V, x_M, x_A, x_B$ need to satisfy the specific conditions given in the problem statement. **Constraints** are mathematical relationships among decision variables or expressions, formulated either as equalities or inequalities. In this problem, for each resource there is a linear constraint that limits overall production:
# 
# $$
# \begin{aligned}
#     10 y_U + 9 y_V  & \leq x_M & & \text{raw material}\\
#     1 y_U + 1 y_V & \leq x_A & &\text{labor A} \\
#     2 y_U + 1 y_V & \leq x_B & & \text{labor B}\\
# \end{aligned}
# $$
# 
# We are now ready to formulate the full mathematical optimization problem in the following canonical way: First we state the objective function to be maximized (or minimized), then we list all the constraints, and lastly we list all the decision variables and their bounds:
# 
# $$
# \begin{align}
# \max \quad & 270 y_U + 210 y_V - 10 x_M - 50 x_A - 40 x_B \\
# \text{such that = s.t.} \quad & 10 y_U + 9 y_V  \leq x_M \nonumber \\
#  & 1 y_U + 1 y_V \leq x_A \nonumber \\
#  & 2 y_U + 1 y_V \leq x_B \nonumber \\
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
# In many textbooks, it is customary to list the decision variables under the $\max$ symbol so as to clearly distinguish between the variables and any other parameters (which might also be expressed as symbols or letters) in the problem formulation. Throughout this website, however, we stick to the convention that the decision variables are only those which explicitly state their domains as part of the problem formulation.
# 
# An **optimal solution** of the problem is any vector of decision variables that meets the constraints and achieves the maximum/minimum objective.
# 
# However, even for a simple problem like this one, it is not immediately clear what the optimal solution is. This is exactly where mathematical optimization algorithms come into play - they are generic procedures that can find the optimal solutions of problems as long as these problems can be formulated in a standardized fashion as above. 
# 
# For a practitioner, mathematical optimization often boils down to formulating the problem as a model above, then passing it over to one of the open-source or commercial software packages that can solve such a model regardless of what was the original *story* behind the model. To do so, we need an interface of communication between the models and the algorithms. In this website, we adopt a Python-based interface which is the ```Pyomo``` modeling package.
# 
# The next step is to create the corresponding Pyomo model, which we will carry out in the next [notebook](production-planning-basic.ipynb).
