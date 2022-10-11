#!/usr/bin/env python
# coding: utf-8

# # Airline seat allocation problem
# 
# An airlines is trying to decide how to partition a new plane for the Amsterdam-Buenos Aires route. This plane can seat 200 economy class passengers. A section can be created for first class seats but each of these seats takes the space of 2 economy class seats. A business class section can also be created, but each of these seats takes as much space as 1.5 economy class seats. The profit on a first class ticket is, however, three times the profit of an economy ticket, while a business class ticket has a profit of two times an economy ticket's profit. Once the plane is partitioned into these seating classes, it cannot be changed. The airlines knows, however, that the plane will not always be full in each section. They have decided that three scenarios will occur with about the same frequency: 
# 
# (1) weekday morning and evening traffic, 
# 
# (2) weekend traffic, 
# 
# (3) weekday midday traffic. 
# 
# Under Scenario 1, they think they can sell as many as 20 first class tickets, 50 business class tickets, and 200 economy tickets. Under Scenario 2, these figures are $10 , 25 $, and $175$, while under Scenario 3, they are $5 , 10$, and $150$. The following table reports the forecast demand in the three scenarios.
# 
# | Scenario | First class seats | Business class seats | Economy class seats |
# | :-: | :-: | :-: | :-: |
# | Scenario 1 | 20 | 50 | 200 |
# | Scenario 2 | 10 | 25 | 175 |
# | Scenario 3 | 5 | 10 | 150 |
# 
# Despite these estimates, the airline will not sell more tickets than seats in each of the sections (hence no overbooking strategy).
# 
# (a) Implement and solve the extensive form of the stochastic program for the optimal seat allocation aiming to maximize the airline profit.

# In[24]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/jckantor/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_glpk()
helper.install_cbc()
helper.install_ipopt()


# In[27]:


import pyomo.environ as pyo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import logging
from IPython.display import Markdown
from io import StringIO

cbc_solver = pyo.SolverFactory('cbc')
glpk_solver = pyo.SolverFactory('glpk')
ipopt_solver = pyo.SolverFactory('ipopt')


# ## Model

# In[3]:


model = pyo.ConcreteModel()

model.classes = pyo.Set(initialize=['F', 'B', 'E'])
model.totalseats = 200
model.pricefactor_F = 3.0
model.pricefactor_B = 2.0
model.seatfactor_F = 2.0 
model.seatfactor_B = 1.5

# first stage variables
model.seats = pyo.Var(model.classes, within=pyo.NonNegativeIntegers) 

# first stage constraint
model.equivalentseatsF = pyo.Expression(expr=model.seats['F']*model.seatfactor_F)
model.equivalentseatsB = pyo.Expression(expr=model.seats['B']*model.seatfactor_B)
model.equivalentseatsE = pyo.Expression(expr=model.seats['E'])
model.planeseats = pyo.Constraint(expr=model.equivalentseatsF + model.equivalentseatsB + model.equivalentseatsE <= model.totalseats)

model.scenarios = pyo.Set(initialize=[1,2,3])

# second stage variables (labelled as 1,2,3 depending on the scenario)
model.sell = pyo.Var(model.classes, model.scenarios, within=pyo.NonNegativeIntegers)

# second stage constraints
model.demandF_1 = pyo.Constraint(expr= model.sell['F',1] <= 20)
model.limitF_1 = pyo.Constraint(expr= model.sell['F',1] <= model.seats['F'])
model.demandF_2 = pyo.Constraint(expr= model.sell['F',2] <= 10)
model.limitF_2 = pyo.Constraint(expr= model.sell['F',2] <= model.seats['F'])
model.demandF_3 = pyo.Constraint(expr= model.sell['F',3] <= 5)
model.limitF_3 = pyo.Constraint(expr= model.sell['F',3] <= model.seats['F'])
model.demandB_1 = pyo.Constraint(expr= model.sell['B',1] <= 50)
model.limitB_1 = pyo.Constraint(expr= model.sell['B',1] <= model.seats['B'])
model.demandB_2 = pyo.Constraint(expr= model.sell['B',2] <= 25)
model.limitB_2 = pyo.Constraint(expr= model.sell['B',2] <= model.seats['B'])
model.demandB_3 = pyo.Constraint(expr= model.sell['B',3] <= 10)
model.limitB_3 = pyo.Constraint(expr= model.sell['B',3] <= model.seats['B'])
model.demandE_1 = pyo.Constraint(expr= model.sell['E',1] <= 200)
model.limitE_1 = pyo.Constraint(expr= model.sell['E',1] <= model.seats['E'])
model.demandE_2 = pyo.Constraint(expr= model.sell['E',2] <= 175)
model.limitE_2 = pyo.Constraint(expr= model.sell['E',2] <= model.seats['E'])
model.demandE_3 = pyo.Constraint(expr= model.sell['E',3] <= 150)
model.limitE_3 = pyo.Constraint(expr= model.sell['E',3] <= model.seats['E'])

def second_stage_profit(model):
    total_1 = model.sell['F',1] * model.pricefactor_F + model.sell['B',1] * model.pricefactor_B + model.sell['E',1]
    total_2 = model.sell['F',2] * model.pricefactor_F + model.sell['B',2] * model.pricefactor_B + model.sell['E',2]
    total_3 = model.sell['F',3] * model.pricefactor_F + model.sell['B',3] * model.pricefactor_B + model.sell['E',3]
    return (total_1 + total_2 + total_3)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(seat allocation) $x_F = {model.seats['F'].value:.0f}$, $e_B = {model.seats['B'].value:.0f}$, $e_E = {model.seats['E'].value:.0f}$"))
display(Markdown(f"(equivalent seat allocation) $e_F = {model.equivalentseatsF.expr():.0f}$, $e_B = {model.equivalentseatsB.expr():.0f}$, $e_E = {model.equivalentseatsE.expr():.0f}$"))
display(Markdown(f"(recourse sell action scenario 1) $s_F = {model.sell['F',1].value:.0f}$, $s_B = {model.sell['B',1].value:.0f}$, $s_E = {model.sell['E',1].value:.0f}$"))
display(Markdown(f"(recourse sell action scenario 2) $s_F = {model.sell['F',2].value:.0f}$, $s_B = {model.sell['B',2].value:.0f}$, $s_E = {model.sell['E',2].value:.0f}$"))
display(Markdown(f"(recourse sell action scenario 3) $s_F = {model.sell['F',3].value:.0f}$, $s_B = {model.sell['B',3].value:.0f}$, $s_E = {model.sell['E',3].value:.0f}$"))
display(Markdown(f"**Optimal objective value:** ${model.total_expected_profit():.0f}$ (in units of economy ticket price)"))


# In[31]:


# seat data
seat_data = {
    "F": {"price factor": 3.0, "seat factor": 2.0},
    "B": {"price factor": 2.0, "seat factor": 1.5},
    "E": {"price factor": 1.0, "seat factor": 1.0},
}

# scenario data
scenario_data = {
    1: {'F': 20, 'B': 50, 'E': 200},
    2: {'F': 10, 'B': 25, 'E': 175},
    3: {'F':  5, 'B': 10, 'E': 150}
}


# In[31]:


seat_data = """
class, price factor, seat factor
F, 3.0, 2.0
B, 2.0, 1.5
E, 1.0, 1.0
"""

seat_data = pd.read_csv(StringIO(seat_data), index_col="class")
display(seat_data)

scenario_data = """
F, B, E
20, 50, 200
10, 25, 175
5, 10, 150
"""

scenario_data = pd.read_csv(StringIO(scenario_data))
display(scenario_data)


# In[33]:


model = pyo.ConcreteModel()

model.total_seats = pyo.Param(initialize=200)

model.classes = pyo.Set(initialize=seat_data.index)

@model.Param(model.classes)
def price_factor(model, c):
    return seat_data.loc[c, "price factor"]

@model.Param(model.classes)
def seat_factor(model, c):
    return seat_data.loc[c, "seat factor"]  

# first stage variables
model.seats = pyo.Var(model.classes, domain=pyo.NonNegativeIntegers) 

@model.Expression(model.classes)
def equivalent_seats(model, c):
    return model.seats[c] * model.seat_factor[c]

@model.Constraint()
def plane_seats(model):
    return sum(model.equivalent_seats[c] for c in model.classes) <= model.total_seats

model.scenarios = pyo.Set(initialize=scenario_data.keys())

# second stage variables (labeled as 1,2,3 depending on the scenario)
model.sell = pyo.Var(model.classes, model.scenarios, domain=pyo.NonNegativeIntegers)

# second stage constraints
@model.Constraint(model.classes, model.scenarios)
def demand(model, c, s):
    return model.sell[c, s] <= scenario_data[s][c]

@model.Constraint(model.classes, model.scenarios)
def limit(model, c, s):
    return model.sell[c, s] <= model.seats[c]

# objective
@model.Expression()
def second_stage_profit(model):
    total =  sum(sum(model.sell[c, s] * model.price_factor[c] for s in model.scenarios) for c in model.classes)
    return total / len(model.scenarios)

@model.Objective(sense=pyo.maximize)
def total_expected_profit(model):
    return model.second_stage_profit

result = cbc_solver.solve(model)

sout = f"**Optimal objective value:** ${model.total_expected_profit():.0f}$ (in units of economy ticket price) <p>"
sout += "**Solution:** <p>"
sout += "\n\n"
sout += "| Variable " + "".join(f"| ${c}$ " for c in model.classes)
sout += "\n| :--- " + "".join(["| :--: "] * len(model.classes))
sout += "\n| seat allocation" + "".join(f"| ${model.seats[c]():.0f}$" for c in model.classes)
sout += "\n| equivalent seat allocation" + "".join(f"| ${model.equivalent_seats[c].expr():.0f}$" for c in model.classes)
for s in model.scenarios:
    sout += f"\n| recourse sell action scenario {s}"
    sout += "".join(f"| ${model.sell[c, s]():.0f}$" for c in model.classes)
    
display(Markdown(sout))


# Assume now that the airline wishes a special guarantee for its clients enrolled in its loyalty program. In particular, it wants $98\%$ probability to cover the demand of first-class seats and $95\%$ probability to cover the demand of business class seats (by clients of the loyalty program). First-class passengers are covered if they get a first-class seat. Business class passengers are covered if they get either a business or a first-class seat (upgrade). Assume weekday demands of loyalty-program passengers are normally distributed, say $\xi_F \sim \mathcal N(16,16)$ and $\xi_B \sim \mathcal N(30,48)$ for first-class and business, respectively. Also assume that the demands for first-class and business class seats are independent.
# Let $x_1$ be the number of first-class seats and $x_2$ the number of business seats. The probabilistic constraints are simply
# 
# $$
# 	\mathbb P(x_1 \geq \xi_F ) \geq 0.98, \qquad \text{ and } \qquad \mathbb P(x_1 +x_2 \geq \xi_F + \xi_B ) \geq 0.95.
# $$
# 
# In Exercise 3 of the tutorial you rewrote these equivalently as linear constraints, specifically 
# 
# $$
# 	(x_1 - 16)/\sqrt{16} \geq 2.054 \qquad \text{ and } \qquad (x_1 +x_2 - 46)/ \sqrt{64} \geq 1.645.
# $$
# 
# (b) Add to your implementation of the extensive form the two equivalent deterministic constraints corresponding to the two chance constraints and find the new optimal solution meeting these additional constraints. How is it different from the previous one?

# In[6]:


model = pyo.ConcreteModel()

model.classes = pyo.Set(initialize=['F', 'B', 'E'])
model.totalseats = 200
model.pricefactor_F = 3.0
model.pricefactor_B = 2.0
model.seatfactor_F = 2.0 
model.seatfactor_B = 1.5

# first stage variables
model.seats = pyo.Var(model.classes, within=pyo.NonNegativeIntegers) 

# first stage constraint
model.equivalentseatsF = pyo.Expression(expr=model.seats['F']*model.seatfactor_F)
model.equivalentseatsB = pyo.Expression(expr=model.seats['B']*model.seatfactor_B)
model.equivalentseatsE = pyo.Expression(expr=model.seats['E'])
model.planeseats = pyo.Constraint(expr=model.equivalentseatsF + model.equivalentseatsB + model.equivalentseatsE <= model.totalseats)
model.loyaltyF = pyo.Constraint(expr = model.seats['F'] >= 24.22) #loyalty constraint 1
model.loyaltyFB = pyo.Constraint(expr = model.seats['F']+model.seats['B'] >= 59.16)  #loyalty constraint 2

model.scenarios = pyo.Set(initialize=[1,2,3])

# second stage variables (labelled as 1,2,3 depending on the scenario)
model.sell = pyo.Var(model.classes, model.scenarios, within=pyo.NonNegativeIntegers)

# second stage constraints
model.demandF_1 = pyo.Constraint(expr= model.sell['F',1] <= 20)
model.limitF_1 = pyo.Constraint(expr= model.sell['F',1] <= model.seats['F'])
model.demandF_2 = pyo.Constraint(expr= model.sell['F',2] <= 10)
model.limitF_2 = pyo.Constraint(expr= model.sell['F',2] <= model.seats['F'])
model.demandF_3 = pyo.Constraint(expr= model.sell['F',3] <= 5)
model.limitF_3 = pyo.Constraint(expr= model.sell['F',3] <= model.seats['F'])
model.demandB_1 = pyo.Constraint(expr= model.sell['B',1] <= 50)
model.limitB_1 = pyo.Constraint(expr= model.sell['B',1] <= model.seats['B'])
model.demandB_2 = pyo.Constraint(expr= model.sell['B',2] <= 25)
model.limitB_2 = pyo.Constraint(expr= model.sell['B',2] <= model.seats['B'])
model.demandB_3 = pyo.Constraint(expr= model.sell['B',3] <= 10)
model.limitB_3 = pyo.Constraint(expr= model.sell['B',3] <= model.seats['B'])
model.demandE_1 = pyo.Constraint(expr= model.sell['E',1] <= 200)
model.limitE_1 = pyo.Constraint(expr= model.sell['E',1] <= model.seats['E'])
model.demandE_2 = pyo.Constraint(expr= model.sell['E',2] <= 175)
model.limitE_2 = pyo.Constraint(expr= model.sell['E',2] <= model.seats['E'])
model.demandE_3 = pyo.Constraint(expr= model.sell['E',3] <= 150)
model.limitE_3 = pyo.Constraint(expr= model.sell['E',3] <= model.seats['E'])

def second_stage_profit(model):
    total_1 = model.sell['F',1] * model.pricefactor_F + model.sell['B',1] * model.pricefactor_B + model.sell['E',1]
    total_2 = model.sell['F',2] * model.pricefactor_F + model.sell['B',2] * model.pricefactor_B + model.sell['E',2]
    total_3 = model.sell['F',3] * model.pricefactor_F + model.sell['B',3] * model.pricefactor_B + model.sell['E',3]
    return (total_1 + total_2 + total_3)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(seat allocation) $x_F = {model.seats['F'].value:.0f}$, $e_B = {model.seats['B'].value:.0f}$, $e_E = {model.seats['E'].value:.0f}$"))
display(Markdown(f"(equivalent seat allocation) $e_F = {model.equivalentseatsF.expr():.0f}$, $e_B = {model.equivalentseatsB.expr():.0f}$, $e_E = {model.equivalentseatsE.expr():.0f}$"))
display(Markdown(f"(recourse sell action scenario 1) $s_F = {model.sell['F',1].value:.0f}$, $s_B = {model.sell['B',1].value:.0f}$, $s_E = {model.sell['E',1].value:.0f}$"))
display(Markdown(f"(recourse sell action scenario 2) $s_F = {model.sell['F',2].value:.0f}$, $s_B = {model.sell['B',2].value:.0f}$, $s_E = {model.sell['E',2].value:.0f}$"))
display(Markdown(f"(recourse sell action scenario 3) $s_F = {model.sell['F',3].value:.0f}$, $s_B = {model.sell['B',3].value:.0f}$, $s_E = {model.sell['E',3].value:.0f}$"))
display(Markdown(f"**Optimal objective value:** ${model.total_expected_profit():.0f}$ (in units of economy ticket price)"))


# Assume now that the ticket demand for the three categories is captured by a $3$-dimensional multivariate normal with mean $\mu=(16,30,180)$ and covariance matrix 
# $$
# \Sigma= \left(
# \begin{array}{ccc}
#  3.5 & 3.7 & 2.5 \\
#  3.7 & 6.5 & 7.5 \\
#  2.5 & 7.5 & 25.2 \\
# \end{array}
# \right).
# $$
# 
# (c) Solve approximately the airline seat allocation problem (with the loyalty constraints) using the Sample Average Approximation method. More specifically, sample $N=1000$ points from the multivariate normal distribution and solve the extensive form for the stochastic LP resulting from those $N=1000$ scenarios.

# In[5]:


def AirlineSAA(N, sample):

    model = pyo.ConcreteModel()

    def indices_rule(model):
        return range(N)

    model.scenarios = pyo.Set(initialize=indices_rule)
    model.demandF = pyo.Param(model.scenarios, initialize=dict(enumerate(sample[:,0])))
    model.demandB = pyo.Param(model.scenarios, initialize=dict(enumerate(sample[:,1])))
    model.demandE = pyo.Param(model.scenarios, initialize=dict(enumerate(sample[:,2])))

    model.classes = pyo.Set(initialize=['F', 'B', 'E'])
    model.totalseats = 200
    model.pricefactor_F = 3.0
    model.pricefactor_B = 2.0
    model.seatfactor_F = 2.0 
    model.seatfactor_B = 1.5

    # first stage variables
    model.seats = pyo.Var(model.classes, within=pyo.NonNegativeIntegers) 

    # first stage constraint
    model.equivalentseatsF = pyo.Expression(expr=model.seats['F']*model.seatfactor_F)
    model.equivalentseatsB = pyo.Expression(expr=model.seats['B']*model.seatfactor_B)
    model.equivalentseatsE = pyo.Expression(expr=model.seats['E'])
    model.planeseats = pyo.Constraint(expr=model.equivalentseatsF + model.equivalentseatsB + model.equivalentseatsE <= model.totalseats)
    model.loyaltyF = pyo.Constraint(expr = model.seats['F'] >= 24.22)
    model.loyaltyFB = pyo.Constraint(expr = model.seats['F']+model.seats['B'] >= 59.16)

    # second stage variables
    model.sell = pyo.Var(model.classes, model.scenarios, within=pyo.NonNegativeIntegers)

    # second stage constraints
    model.demandFlim = pyo.ConstraintList()
    model.limitF = pyo.ConstraintList()
    model.demandBlim = pyo.ConstraintList()
    model.limitB = pyo.ConstraintList()
    model.demandElim = pyo.ConstraintList()
    model.limitE = pyo.ConstraintList()
    for i in model.scenarios:
        model.demandFlim.add(expr= model.sell['F',i] <= model.demandF[i])
        model.limitF.add(expr= model.sell['F',i] <= model.seats['F'])
        model.demandBlim.add(expr= model.sell['B',i] <= model.demandB[i])
        model.limitB.add(expr= model.sell['B',i] <= model.seats['B'])
        model.demandElim.add(expr= model.sell['E',i] <= model.demandE[i])
        model.limitE.add(expr= model.sell['E',i] <= model.seats['E'])

    def second_stage_profit(model):
        return sum([model.sell['F',i] * model.pricefactor_F + model.sell['B',i] * model.pricefactor_B + model.sell['E',i] for i in model.scenarios])/float(N)

    model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

    def total_profit(model):
        return model.second_stage_profit

    model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

    result = cbc_solver.solve(model)
    display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
    display(Markdown(f"**Solution:**"))
    display(Markdown(f"(seat allocation) $x_F = {model.seats['F'].value:.0f}$, $e_B = {model.seats['B'].value:.0f}$, $e_E = {model.seats['E'].value:.0f}$"))
    display(Markdown(f"(equivalent seat allocation) $e_F = {model.equivalentseatsF.expr():.0f}$, $e_B = {model.equivalentseatsB.expr():.0f}$, $e_E = {model.equivalentseatsE.expr():.0f}$"))
    display(Markdown(f"**Optimal objective value:** ${model.total_expected_profit():.0f}$ (in units of economy ticket price)"))

N = 1000
np.random.seed(1)
samples = np.random.multivariate_normal([16, 30, 180],[[3.5, 3.7, 2.5],[3.7, 6.5, 7.5],[2.5, 7.5, 25.2]], N)
AirlineSAA(N, samples)

