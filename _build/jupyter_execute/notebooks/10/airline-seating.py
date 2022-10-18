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

# In[1]:


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


# In[4]:


import pyomo.environ as pyo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from IPython.display import Markdown

cbc_solver = pyo.SolverFactory('cbc')
glpk_solver = pyo.SolverFactory('glpk')
ipopt_solver = pyo.SolverFactory('ipopt')


# ## Model

# In[6]:


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


# In[29]:


def airline():

    m = pyo.ConcreteModel()

    m.classes = pyo.Set(initialize=seat_data.keys())
    m.scenarios = pyo.Set(initialize=scenario_data.keys())

    m.total_seats = pyo.Param(initialize=200)

    @m.Param(m.classes)
    def price_factor(m, c):
        return seat_data[c]["price factor"]

    @m.Param(m.classes)
    def seat_factor(m, c):
        return seat_data[c]["seat factor"]  

    # first stage variables
    m.seats = pyo.Var(m.classes, domain=pyo.NonNegativeIntegers) 

    @m.Expression(model.classes)
    def equivalent_seats(m, c):
        return m.seats[c] * m.seat_factor[c]

    @m.Constraint()
    def plane_seats(m):
        return sum(m.equivalent_seats[c] for c in m.classes) <= m.total_seats

    # second stage variables
    m.sell = pyo.Var(m.classes, m.scenarios, domain=pyo.NonNegativeIntegers)

    # second stage constraints
    @m.Constraint(m.classes, m.scenarios)
    def demand(m, c, s):
        return m.sell[c, s] <= scenario_data[s][c]

    @m.Constraint(m.classes, m.scenarios)
    def limit(m, c, s):
        return m.sell[c, s] <= m.seats[c]

    # objective
    @m.Expression()
    def second_stage_profit(m):
        total =  sum(sum(m.sell[c, s] * m.price_factor[c] for s in m.scenarios) for c in m.classes)
        return total / len(m.scenarios)

    @m.Objective(sense=pyo.maximize)
    def total_expected_profit(m):
        return m.second_stage_profit
    
    return m


m = airline()
result = pyo.SolverFactory('cbc').solve(m)

sout = f"**Optimal objective value:** ${m.total_expected_profit():.0f}$ (in units of economy ticket price) <p>"
sout += "**Solution:** <p>"
sout += "\n\n"
sout += "| Variable " + "".join(f"| ${c}$ " for c in m.classes)
sout += "\n| :--- " + "".join(["| :--: "] * len(m.classes))
sout += "\n| seat allocation" + "".join(f"| ${m.seats[c]():.0f}$" for c in m.classes)
sout += "\n| equivalent seat allocation" + "".join(f"| ${m.equivalent_seats[c].expr():.0f}$" for c in m.classes)
for s in model.scenarios:
    sout += f"\n| recourse sell action scenario {s}"
    sout += "".join(f"| ${m.sell[c, s]():.0f}$" for c in model.classes)
    
display(Markdown(sout))


# ## Chance Constraints
# 
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

# In[32]:


def airline_loyalty():

    m = airline()
    m.loyaltyF = pyo.Constraint(expr = m.seats['F'] >= 24.22) #loyalty constraint 1
    m.loyaltyFB = pyo.Constraint(expr = m.seats['F'] + m.seats['B'] >= 59.16)  #loyalty constraint 2
    
    return m

m = airline_loyalty()


result = pyo.SolverFactory('cbc').solve(m)

sout = f"**Optimal objective value:** ${m.total_expected_profit():.0f}$ (in units of economy ticket price) <p>"
sout += "**Solution:** <p>"
sout += "\n\n"
sout += "| Variable " + "".join(f"| ${c}$ " for c in m.classes)
sout += "\n| :--- " + "".join(["| :--: "] * len(m.classes))
sout += "\n| seat allocation" + "".join(f"| ${m.seats[c]():.0f}$" for c in m.classes)
sout += "\n| equivalent seat allocation" + "".join(f"| ${m.equivalent_seats[c].expr():.0f}$" for c in m.classes)
for s in model.scenarios:
    sout += f"\n| recourse sell action scenario {s}"
    sout += "".join(f"| ${m.sell[c, s]():.0f}$" for c in model.classes)
    
display(Markdown(sout))


# ## Sample Average
# 
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

# In[33]:


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
samples = np.random.multivariate_normal([16, 30, 180], [[3.5, 3.7, 2.5], [3.7, 6.5, 7.5], [2.5, 7.5, 25.2]], N)
AirlineSAA(N, samples)


# In[ ]:




