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
helper.install_cbc()


# In[2]:


import pyomo.environ as pyo
import numpy as np
import pandas as pd


# ## Model

# In[3]:


# seat data
seat_data = pd.DataFrame({
    "F": {"price factor": 3.0, "seat factor": 2.0},
    "B": {"price factor": 2.0, "seat factor": 1.5},
    "E": {"price factor": 1.0, "seat factor": 1.0},
}).T

# scenario data
scenario_data = pd.DataFrame({
    1: {'F': 20, 'B': 50, 'E': 200},
    2: {'F': 10, 'B': 25, 'E': 175},
    3: {'F':  5, 'B': 10, 'E': 150}
}).T

display(seat_data)
display(scenario_data)


# In[4]:


def airline(total_seats, seat_data, scenario_data):

    m = pyo.ConcreteModel()
    
    m.classes = pyo.Set(initialize=seat_data.index)
    m.scenarios = pyo.Set(initialize=scenario_data.index)

    # first stage variables and constraints
    
    m.seats = pyo.Var(m.classes, domain=pyo.NonNegativeIntegers) 

    @m.Expression(m.classes)
    def equivalent_seats(m, c):
        return m.seats[c] * seat_data.loc[c, "seat factor"]

    @m.Constraint()
    def plane_seats(m):
        return sum(m.equivalent_seats[c] for c in m.classes) <= total_seats

    # second stage variables and constraints
    
    m.sell = pyo.Var(m.classes, m.scenarios, domain=pyo.NonNegativeIntegers)

    @m.Constraint(m.classes, m.scenarios)
    def demand(m, c, s):
        return m.sell[c, s] <= scenario_data.loc[s, c]

    @m.Constraint(m.classes, m.scenarios)
    def sell_limit(m, c, s):
        return m.sell[c, s] <= m.seats[c]

    # objective
    
    @m.Expression()
    def second_stage_profit(m):
        total =  sum(sum(m.sell[c, s] * seat_data.loc[c, "price factor"] for s in m.scenarios) for c in m.classes)
        return total / len(m.scenarios)

    @m.Objective(sense=pyo.maximize)
    def total_expected_profit(m):
        return m.second_stage_profit
    
    return m

def report_seats(m):
    print(f"Optimal objective value in units of economy ticket price: {m.total_expected_profit():.0f}")
    seats = {}
    seats["seat allocation"] =  {c: m.seats[c]() for c in m.classes}
    seats["equivalent seat allocation"] = {c: m.equivalent_seats[c].expr() for c in m.classes}
    display(pd.DataFrame(seats).T)
    
def report_scenarios(m):
    sales = {}
    for s in m.scenarios:
        sales[f"recourse sell action scenario {s}"] = {c: m.sell[c, s]() for c in m.classes} 
    display(pd.DataFrame(sales).T)
    
m = airline(200, seat_data, scenario_data)
pyo.SolverFactory('cbc').solve(m)
report_seats(m)
report_scenarios(m)


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

# In[5]:


def airline_loyalty(total_seats, seat_data, scenario_data):
    m = airline(total_seats, seat_data, scenario_data)
    m.loyaltyF = pyo.Constraint(expr = m.seats['F'] >= 24.22)
    m.loyaltyFB = pyo.Constraint(expr = m.seats['F'] + m.seats['B'] >= 59.16)
    return m

m = airline_loyalty(200, seat_data, scenario_data)
pyo.SolverFactory('cbc').solve(m)
report_seats(m)
report_scenarios(m)


# ## Sample Average
# 
# Assume now that the ticket demand for the three categories is captured by a $3$-dimensional multivariate normal with mean $\mu=(16,30,180)$ and covariance matrix 
# 
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

# In[6]:


N = 1000
np.random.seed(1)
samples = np.random.multivariate_normal([16, 30, 180], [[3.5, 3.7, 2.5], [3.7, 6.5, 7.5], [2.5, 7.5, 25.2]], N)

saa_data = pd.DataFrame(samples, columns=seat_data.index)
m = airline_loyalty(200, seat_data, saa_data)
pyo.SolverFactory('cbc').solve(m)
report_seats(m)


# In[ ]:




