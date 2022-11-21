#!/usr/bin/env python
# coding: utf-8

# # Farmer's problem and some of its variants
# 
# The [Farmer's Problem](https://www.math.uh.edu/~rohop/Spring_15/Chapter1.pdf) is a teaching example presented in the well-known textbook by John Birge and Francois Louveaux 
# 
# * Birge, John R., and Francois Louveaux. Introduction to stochastic programming. Springer Science & Business Media, 2011.
# 
# This problem widely used in teaching notes and presentations. Here we present a Pyomo solution to this problem and several variants.

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import logging
from IPython.display import Markdown

cbc_solver = pyo.SolverFactory('cbc')
glpk_solver = pyo.SolverFactory('glpk')
ipopt_solver = pyo.SolverFactory('ipopt')


# ## Base Model
# 
# In the [farmer's problem](https://www.math.uh.edu/~rohop/Spring_15/Chapter1.pdf) a farmer has to allocate $500$ acres of land to three different types of crops aiming to maximize her profit. 
# 
# Recall that:
# 
# * Planting one acre of wheat, corn and beet costs 150, 230 and 260 euro, respectively.
# 
# * At least 200 tons of wheat and 240 tons of corn are needed for cattle feed, which can be purchased from a wholesaler if not harvested by her farm.
# 
# * Up to 6,000 tons of sugar beets can be sold for 36 euro per ton, while any additional amounts can be sold for 10 euro per ton.
# 
# * Any wheat or corn not used for the cattle can be sold at 170 euro and 150 euro per ton of wheat and corn, respectively. The wholesaler sells the wheat or corn at a higher price, namely 238 euro and 210 euro per ton, respectively.
# 
# In her decision, the farmer considers three weather scenarios, each one having a different yield in tons/acre per crop type as summarized by the following table.
# 
# | Scenario | Yield for wheat <br> (tons/acre)| Yield for corn <br> (tons/acre) | Yield for beets <br> (tons/acre) |
# | :-- | :-: | :-: | :-: |
# | Good weather | 3 | 3.6 | 24 |
# | Average weather | 2.5 | 3 | 20 |
# | Bad weather | 2 | 2.4 | 16 |
# 
# We first consider the case in which all the prices are fixed and not weather-dependent. The following table summarizes the data.
# 
# | Commodity | Sell Price <br> (euro/ton) | Market <br> Demand <br> (tons) | Purchase <br> Price <br> (euro/ton) | Cattle Feed <br> Required <br> (tons) | Planting <br> Cost <br> (euro/acre) |
# | :-- | :--: | :--: | :--: | :--: | :--: |
# | Wheat | 170 | - | 238 | 200 | 150 |
# | Corn | 150 | - | 210 | 240 | 230 |
# | Beets | 36 | 6000 | - | 0 | 260 | 6000 |
# | Beets extra | 10 | - | - | 0 | 260 |
# 
# (a) Implement the extensive form of stochastic LP corresponding to the farmer's problem in Pyomo and solve it.

# In[204]:


M = 10000

data = """
crop,        planting_cost, crop_yield, sell_price, demand, buy_price, cattle_feed
wheat,                 150,        2.5,        170,       ,       238,         200
corn,                  230,          3,        150,       ,       210,         240  
beets,                 260,         20,         36,   6000,          ,           0
beets extra,           260,         
"""

# constant parameters
planting_cost = {"W": 150, "C": 230, "B": 260, "B extra": 260}
crop_yield = {"W": 2.5, "C": 3, "B": 20, "B extra": 20}
sell_price = {"W": 170, "C": 150, "B": 36, "B extra": 10}
demand = {"W": M, "C": M, "B": 6000, "B extra": M}
buy_price = {"W": 238, "C": 210, "B": M, "B extra": M}
cattle_feed = {"W": 200, "C": 240, "B": 0, "B extra": 0}

# scenario dependent parameters
yield_factor = {"H": 1.2, "M": 1.0, "L": 0.8}


# In[205]:


m = pyo.ConcreteModel()

# mutable parameter
m.total_acres = pyo.Param(initialize=500, mutable=True)

# sets
m.crops = pyo.Set(initialize=planting_cost.keys())
m.scenarios = pyo.Set(initialize=yield_factor.keys())

# constant parameter values
m.planting_cost = pyo.Param(m.crops, initialize=planting_cost)
m.crop_yield = pyo.Param(m.crops, initialize=crop_yield)
m.sell_price = pyo.Param(m.crops, initialize=sell_price)
m.buy_price = pyo.Param(m.crops, initialize=buy_price)
m.demand = pyo.Param(m.crops, initialize=demand)
m.cattle_feed = pyo.Param(m.crops, initialize=cattle_feed)

# scenario dependent parameter values
m.yield_factor = pyo.Param(m.scenarios, initialize=yield_factor)

# first stage variables

m.plant = pyo.Var(m.crops, domain=pyo.NonNegativeReals)

# first stage constraint
@m.Constraint()
def acres(m):
    return sum(m.plant[c] for c in m.crops) <= m.total_acres

# first stage profit
@m.Expression()
def first_stage_profit(m, c):
    return -sum(m.plant[c] * m.planting_cost[c] for c in m.crops)

# second stage variables
m.produced = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.buy = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.sell = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)

# second stage constraints
@m.Constraint(m.crops, m.scenarios)
def crops_produced(m, c, s):
    return m.produced[c, s] == m.plant[c] * m.crop_yield[c] * m.yield_factor[s]

@m.Constraint(m.crops, m.scenarios)
def demand_limit(m, c, s):
    return m.sell[c, s] <= m.demand[c]

@m.Constraint(m.crops, m.scenarios)
def balance(m, c, s):
    return m.produced[c, s] + m.buy[c, s] == m.cattle_feed[c] + m.sell[c, s]

# second stage profit
@m.Expression(m.scenarios)
def second_stage_profit(m, s):
    revenue = sum(m.sell[c, s] * m.sell_price[c] for c in m.crops) 
    expense = sum(m.buy[c, s] * m.buy_price[c] for c in m.crops)
    return revenue - expense

# Objective
@m.Objective(sense=pyo.maximize)
def total_profit(m):
    return m.first_stage_profit + sum(m.second_stage_profit[s] for s in m.scenarios) / len(m.scenarios)

pyo.SolverFactory('cbc').solve(m)
    
print(f"\nFirst Stage Profit: {m.first_stage_profit.expr():0.2f} euros")
print("\tCrop       Planted")
print("\t           (acres)")
for c in m.crops:
    print(f"\t{c:8s}   {m.plant[c].value:7.1f}")
    
print(f"\nSecond Stage / Recourse Solutions")
for s in m.scenarios:
    print(f"\nScenario {s} Profit: {m.second_stage_profit[s].expr():0.2f} euros")
    print("\tCrop      Produced       Buy      Feed      Sell")
    print("\t            (tons)     (tons)    (tons)   (tons)")
    for c in m.crops:
        sout = f"\t{c:8s}"
        sout += f"   {m.produced[c, s].value:7.1f}"
        sout += f"   {m.buy[c, s].value:7.1f}"
        sout += f"   {cattle_feed[c]:7.1f}"
        sout += f"   {m.sell[c, s].value:7.1f}"
        print(sout)

print(f"\nExpected profit = {m.total_profit():0.2f} euros")


# ## Scenario Dependent Prices

# In[202]:


# second stage constraints
@model.Constraint(model.scenarios)
def feed_cattle_W(model, s):
    return model.plant['W'] * 2.5 * model.factor_H - model.sell_H['W', s] + model.buy_H['W', s] >= 200

model.feed_cattle_C_H = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_H - model.sell_H['C'] + model.buy_H['C'] >= 240)
model.sell_S_extra_H = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_H >= model.sell_H['S'] + model.sell_extra_H)
model.sell_S_H = pyo.Constraint(expr=model.sell_H['S'] <= 6000)
model.nobuy_H = pyo.Constraint(expr=model.buy_H['S'] == 0)

model.feed_cattle_W_M = pyo.Constraint(expr=model.plant['W'] * 2.5 - model.sell_M['W'] + model.buy_M['W'] >= 200)
model.feed_cattle_C_M = pyo.Constraint(expr=model.plant['C'] * 3 - model.sell_M['C'] + model.buy_M['C'] >= 240)
model.sell_S_extra_M = pyo.Constraint(expr=model.plant['S'] * 20 >= model.sell_M['S'] + model.sell_extra_M)
model.sell_S_M = pyo.Constraint(expr=model.sell_M['S'] <= 6000)
model.nobuy_M = pyo.Constraint(expr=model.buy_M['S'] == 0)

model.feed_cattle_W_L = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
model.feed_cattle_C_L = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
model.sell_S_extra_L = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

def first_stage_profit(model):
    return -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260

model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

def second_stage_profit(model):
    total_H = -model.buy_H['W'] * 238 - model.buy_H['C'] * 210 + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 + model.sell_H['C'] * 150
    total_M = -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150
    total_L = -model.buy_L['W'] * 238 - model.buy_L['C'] * 210 + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 + model.sell_L['C'] * 150
    return (total_H + total_M + total_L)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.first_stage_profit + model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(land allocation) $x_1 = {model.plant['W'].value:.1f}$, $x_2 = {model.plant['C'].value:.1f}$, $x_3 = {model.plant['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action high yield) $w_1 = {model.sell_H['W'].value:.1f}$, $w_2 = {model.sell_H['C'].value:.1f}$, $w_3 = {model.sell_H['S'].value:.1f}$, $w_4 = {model.sell_extra_H.value:.1f}$"))
display(Markdown(f"(recourse purchase action high yield) $y_1 = {model.buy_H['W'].value:.1f}$, $y_2 = {model.buy_H['C'].value:.1f}$, $y_3 = {model.buy_H['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action medium yield) $w_1 = {model.sell_M['W'].value:.1f}$, $w_2 = {model.sell_M['C'].value:.1f}$, $w_3 = {model.sell_M['S'].value:.1f}$, $w_4 = {model.sell_extra_M.value:.1f}$"))
display(Markdown(f"(recourse purchase action medium yield) $y_1 = {model.buy_M['W'].value:.1f}$, $y_2 = {model.buy_M['C'].value:.1f}$, $y_3 = {model.buy_M['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action low yield) $w_1 = {model.sell_L['W'].value:.1f}$, $w_2 = {model.sell_L['C'].value:.1f}$, $w_3 = {model.sell_L['S'].value:.1f}$, $w_4 = {model.sell_extra_L.value:.1f}$"))
display(Markdown(f"(recourse purchase action low yield) $y_1 = {model.buy_L['W'].value:.1f}$, $y_2 = {model.buy_L['C'].value:.1f}$, $y_3 = {model.buy_L['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.total_expected_profit():.0f}$€"))


# In[9]:


model = pyo.ConcreteModel()

model.crops = pyo.Set(initialize=['W', 'C', 'S'])
model.totalacres = 500
model.factor_H = 1.2 # to obtain the yields in the good weather (high yield) case by multiplying the average ones
model.factor_L = 0.8 # to obtain the yields in the bad weather (low yield) case by multiplying the average ones

# first stage variables
model.plant = pyo.Var(model.crops, domain=pyo.NonNegativeReals) 

# first stage constraint
@model.Constraint()
def total_acres(model):
    return pyo.summation(model.plant) <= model.totalacres

model.scenarios = pyo.Set(initialize=['H', 'M', 'L'])  # high, medium, and low yield scenarios

# second stage variables (labelled as H, M, L depending on the scenario)
# the sell_extra variables refer to the amount of beets to be sold beyond the 6000 threshold, if any

model.sell_H = pyo.Var(model.crops, domain=pyo.NonNegativeReals)
model.buy_H = pyo.Var(model.crops, domain=pyo.NonNegativeReals)
model.sell_extra_H = pyo.Var(domain=pyo.NonNegativeReals) 

model.sell_M = pyo.Var(model.crops, domain=pyo.NonNegativeReals)
model.buy_M = pyo.Var(model.crops, domain=pyo.NonNegativeReals)
model.sell_extra_M = pyo.Var(domain=pyo.NonNegativeReals)

model.sell_L = pyo.Var(model.crops, domain=pyo.NonNegativeReals)
model.buy_L = pyo.Var(model.crops, domain=pyo.NonNegativeReals)
model.sell_extra_L = pyo.Var(domain=pyo.NonNegativeReals)

# second stage constraints
model.feed_cattle_W_H = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_H - model.sell_H['W'] + model.buy_H['W'] >= 200)
model.feed_cattle_C_H = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_H - model.sell_H['C'] + model.buy_H['C'] >= 240)
model.sell_S_extra_H = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_H >= model.sell_H['S'] + model.sell_extra_H)
model.sell_S_H = pyo.Constraint(expr=model.sell_H['S'] <= 6000)
model.nobuy_H = pyo.Constraint(expr=model.buy_H['S'] == 0)

model.feed_cattle_W_M = pyo.Constraint(expr=model.plant['W'] * 2.5 - model.sell_M['W'] + model.buy_M['W'] >= 200)
model.feed_cattle_C_M = pyo.Constraint(expr=model.plant['C'] * 3 - model.sell_M['C'] + model.buy_M['C'] >= 240)
model.sell_S_extra_M = pyo.Constraint(expr=model.plant['S'] * 20 >= model.sell_M['S'] + model.sell_extra_M)
model.sell_S_M = pyo.Constraint(expr=model.sell_M['S'] <= 6000)
model.nobuy_M = pyo.Constraint(expr=model.buy_M['S'] == 0)

model.feed_cattle_W_L = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
model.feed_cattle_C_L = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
model.sell_S_extra_L = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

def first_stage_profit(model):
    return -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260

model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

def second_stage_profit(model):
    total_H = -model.buy_H['W'] * 238 - model.buy_H['C'] * 210 + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 + model.sell_H['C'] * 150
    total_M = -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150
    total_L = -model.buy_L['W'] * 238 - model.buy_L['C'] * 210 + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 + model.sell_L['C'] * 150
    return (total_H + total_M + total_L)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.first_stage_profit + model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(land allocation) $x_1 = {model.plant['W'].value:.1f}$, $x_2 = {model.plant['C'].value:.1f}$, $x_3 = {model.plant['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action high yield) $w_1 = {model.sell_H['W'].value:.1f}$, $w_2 = {model.sell_H['C'].value:.1f}$, $w_3 = {model.sell_H['S'].value:.1f}$, $w_4 = {model.sell_extra_H.value:.1f}$"))
display(Markdown(f"(recourse purchase action high yield) $y_1 = {model.buy_H['W'].value:.1f}$, $y_2 = {model.buy_H['C'].value:.1f}$, $y_3 = {model.buy_H['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action medium yield) $w_1 = {model.sell_M['W'].value:.1f}$, $w_2 = {model.sell_M['C'].value:.1f}$, $w_3 = {model.sell_M['S'].value:.1f}$, $w_4 = {model.sell_extra_M.value:.1f}$"))
display(Markdown(f"(recourse purchase action medium yield) $y_1 = {model.buy_M['W'].value:.1f}$, $y_2 = {model.buy_M['C'].value:.1f}$, $y_3 = {model.buy_M['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action low yield) $w_1 = {model.sell_L['W'].value:.1f}$, $w_2 = {model.sell_L['C'].value:.1f}$, $w_3 = {model.sell_L['S'].value:.1f}$, $w_4 = {model.sell_extra_L.value:.1f}$"))
display(Markdown(f"(recourse purchase action low yield) $y_1 = {model.buy_L['W'].value:.1f}$, $y_2 = {model.buy_L['C'].value:.1f}$, $y_3 = {model.buy_L['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.total_expected_profit():.0f}$€"))


# Please note a second way to create this model which makes use of `pyomo` `blocks`. For an explanation of `blocks` refer to chapter 8 of the `pyomo` book that you may [download](https://vu.on.worldcat.org/oclc/988749903) from the VU library.
# 
# We leave as an exercise to redo the rest of the questions using `blocks` and see how that may help.
# 
# Note that for b) you will need to provide prices as parameters to the blocks. 
# Note as well that is the resolution below already had done that, instead of typing the prices as numerical constants, then the changes would have been immediate. 

# In[8]:


m = pyo.ConcreteModel()

m.crops = pyo.Set(initialize=['W', 'C', 'S'])
m.totalacres = 500

# first stage variables
m.plant = pyo.Var(m.crops, within=pyo.NonNegativeReals) 

# first stage constraint
m.total_acres = pyo.Constraint(expr=pyo.summation(m.plant) <= m.totalacres)

m.scenarios = pyo.Set(initialize=['H', 'M', 'L'])  # high, medium, and low yield scenarios

# this could be a dataframe, or any other data source
nominal_yields = { 'W' : 2.5, 'C' :   3, 'S' : 20 }
factor_yields  = { 'M' : 1, 'H' : 1.2, 'L' : 0.8 }
m.yields = { s : { c : nominal_yields[c]*factor_yields[s] for c in m.crops } for s in m.scenarios }

def scenario_block(b, s):
  b.yields     = pyo.Param(m.crops,initialize=m.yields[s])
  b.sell       = pyo.Var(m.crops, within=pyo.NonNegativeReals)
  b.buy        = pyo.Var(m.crops, within=pyo.NonNegativeReals)
  b.sell_extra = pyo.Var(within=pyo.NonNegativeReals) 
  b.sell_S     = pyo.Constraint(expr=b.sell['S'] <= 6000)
  b.nobuy      = pyo.Constraint(expr=b.buy['S'] == 0)
  b.profit     = pyo.Expression(expr= -238*b.buy['W'] -210*b.buy['C'] +36*b.sell['S'] + 10*b.sell_extra + 170*b.sell['W'] + 150*b.sell['C'])

m.scenario = pyo.Block( m.scenarios, rule=scenario_block)


# second stage (linking) constraints
m.feed_cattle_W = pyo.Constraint(m.scenarios, rule = lambda m, s : m.plant['W'] * m.scenario[s].yields['W'] - m.scenario[s].sell['W'] + m.scenario[s].buy['W'] >= 200)
m.feed_cattle_C = pyo.Constraint(m.scenarios, rule = lambda m, s : m.plant['C'] * m.scenario[s].yields['C'] - m.scenario[s].sell['C'] + m.scenario[s].buy['C'] >= 240)
m.sell_S_extra  = pyo.Constraint(m.scenarios, rule = lambda m, s : m.plant['S'] * m.scenario[s].yields['S'] >= m.scenario[s].sell['S'] + m.scenario[s].sell_extra )

m.first_stage_profit = pyo.Expression( expr = -150*m.plant["W"] -230*m.plant["C"] -260*m.plant["S"] )
m.total_expected_profit = pyo.Objective( rule = lambda m : m.first_stage_profit + sum(m.scenario[s].profit for s in m.scenarios)/3, sense=pyo.maximize )

result = cbc_solver.solve(m)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(land allocation) $x_1 = {m.plant['W'].value:.1f}$, $x_2 = {m.plant['C'].value:.1f}$, $x_3 = {m.plant['S'].value:.1f}$"))
spellout = { 'H' : 'high', 'M' : 'medium', 'L' : 'low' }
for s in m.scenarios:
  display(Markdown(f"(recourse sell action {spellout[s]} yield) $w_1 = {m.scenario[s].sell['W'].value:.1f}$, $w_2 = {m.scenario[s].sell['C'].value:.1f}$, $w_3 = {m.scenario[s].sell['S'].value:.1f}$, $w_4 = {m.scenario[s].sell_extra.value:.1f}$"))
  display(Markdown(f"(recourse purchase action {spellout[s]} yield) $y_1 = {m.scenario[s].buy['W'].value:.1f}$, $y_2 = {m.scenario[s].buy['C'].value:.1f}$, $y_3 = {m.scenario[s].buy['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${m.total_expected_profit.expr():.0f}$€"))


# If the weather is good and yields are high for the farmer, they are probably so also for many other farmers. The total supply is thus increasing, which will lower the prices. Assume the prices going down by 10% for corn and wheat when the weather is good and going up by 10% when the weather is bad. These changes in prices affect both sales and purchases of corn and wheat, but sugar beet prices are not affected by yields. The following table summarizes the scenario-dependent prices:
# 
# | Scenario | Selling price for weath | Selling price for corn | Purchasing price for weath | Purchasing price for corn | 
# | :-: | :-: | :-: | :-: | :-: |
# | Good weather | 153| 135| 214| 189|
# | Average weather | 170 | 150 | 238 | 210 |
# | Bad weather |187| 165| 262| 231|
# 
# (b) Implement the extensive form of stochastic LP corresponding to the farmer's problem in Pyomo that accounts also for the price changes and solve it.
# 
# 

# In[213]:


# scenario dependent parameters
prices = {
    "W": {"H": 0.9, "M": 1.0, "L": 1.1},
    "C": {"H": 0.9, "M": 1.0, "L": 1.1},
    "B": {"H": 1.0, "M": 1.0, "L": 1.0},
    "B extra": {"H": 1.0, "M": 1.0, "L": 1.0},
}


# In[217]:


m = pyo.ConcreteModel()

# mutable parameter
m.total_acres = pyo.Param(initialize=500, mutable=True)

# sets
m.crops = pyo.Set(initialize=planting_cost.keys())
m.scenarios = pyo.Set(initialize=yield_factor.keys())

# constant parameter values
m.planting_cost = pyo.Param(m.crops, initialize=planting_cost)
m.crop_yield = pyo.Param(m.crops, initialize=crop_yield)
m.sell_price = pyo.Param(m.crops, initialize=sell_price)
m.buy_price = pyo.Param(m.crops, initialize=buy_price)
m.demand = pyo.Param(m.crops, initialize=demand)
m.cattle_feed = pyo.Param(m.crops, initialize=cattle_feed)

# scenario dependent parameter values
m.yield_factor = pyo.Param(m.scenarios, initialize=yield_factor)
@m.Param(m.crops, m.scenarios)
def price_factor(m, c, s):
    return prices[c][s]

# first stage variables

m.plant = pyo.Var(m.crops, domain=pyo.NonNegativeReals)

# first stage constraint
@m.Constraint()
def acres(m):
    return sum(m.plant[c] for c in m.crops) <= m.total_acres

# first stage profit
@m.Expression()
def first_stage_profit(m, c):
    return -sum(m.plant[c] * m.planting_cost[c] for c in m.crops)

# second stage variables
m.produced = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.buy = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.sell = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)

# second stage constraints
@m.Constraint(m.crops, m.scenarios)
def crops_produced(m, c, s):
    return m.produced[c, s] == m.plant[c] * m.crop_yield[c] * m.yield_factor[s]

@m.Constraint(m.crops, m.scenarios)
def demand_limit(m, c, s):
    return m.sell[c, s] <= m.demand[c]

@m.Constraint(m.crops, m.scenarios)
def balance(m, c, s):
    return m.produced[c, s] + m.buy[c, s] == m.cattle_feed[c] + m.sell[c, s]

# second stage profit
@m.Expression(m.scenarios)
def second_stage_profit(m, s):
    revenue = sum(m.sell[c, s] * m.sell_price[c] * m.price_factor[c, s] for c in m.crops) 
    expense = sum(m.buy[c, s] * m.buy_price[c] * m.price_factor[c, s] for c in m.crops)
    return revenue - expense

# Objective
@m.Objective(sense=pyo.maximize)
def total_profit(m):
    return m.first_stage_profit + sum(m.second_stage_profit[s] for s in m.scenarios) / len(m.scenarios)

pyo.SolverFactory('cbc').solve(m)
    
print(f"\nFirst Stage Profit: {m.first_stage_profit.expr():0.2f} euros")
print("\tCrop       Planted")
print("\t           (acres)")
for c in m.crops:
    print(f"\t{c:8s}   {m.plant[c].value:7.1f}")
    
print(f"\nSecond Stage / Recourse Solutions")
for s in m.scenarios:
    print(f"\nScenario {s} Profit: {m.second_stage_profit[s].expr():0.2f} euros")
    print("\tCrop      Produced       Buy      Feed      Sell")
    print("\t            (tons)     (tons)    (tons)   (tons)")
    for c in m.crops:
        sout = f"\t{c:8s}"
        sout += f"   {m.produced[c, s].value:7.1f}"
        sout += f"   {m.buy[c, s].value:7.1f}"
        sout += f"   {cattle_feed[c]:7.1f}"
        sout += f"   {m.sell[c, s].value:7.1f}"
        print(sout)

print(f"\nExpected profit = {m.total_profit():0.2f} euros")


# In[5]:


model = pyo.ConcreteModel()

model.crops = pyo.Set(initialize=['W', 'C', 'S'])
model.totalacres = 500
model.factor_H = 1.2 # to obtain the yields in the good weather (high yield) case by multiplying the average ones
model.factor_L = 0.8 # to obtain the yields in the bad weather (low yield) case by multiplying the average ones
model.pricefactor_H = 0.9 # to obtain the prices in the good weather (high yield) case by multiplying the average ones
model.pricefactor_L = 1.1 # to obtain the prices in the bad weather (low yield) case by multiplying the average ones

# first stage variables
model.plant = pyo.Var(model.crops, within=pyo.NonNegativeReals) 

# first stage constraint
model.total_acres = pyo.Constraint(expr=pyo.summation(model.plant) <= model.totalacres)

model.scenarios = pyo.Set(initialize=['H', 'M', 'L'])  # high, medium, and low yield scenarios

# second stage variables (labelled as H,M,L depending on the scenario)
# the sell_extra variables refer to the amount of beets to be sold beyond the 6000 threshold, if any

model.sell_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_H = pyo.Var(within=pyo.NonNegativeReals) 

model.sell_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_M = pyo.Var(within=pyo.NonNegativeReals)

model.sell_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_L = pyo.Var(within=pyo.NonNegativeReals)

# second stage constraints
model.feed_cattle_W_H = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_H - model.sell_H['W'] + model.buy_H['W'] >= 200)
model.feed_cattle_C_H = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_H - model.sell_H['C'] + model.buy_H['C'] >= 240)
model.sell_S_extra_H = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_H >= model.sell_H['S'] + model.sell_extra_H)
model.sell_S_H = pyo.Constraint(expr=model.sell_H['S'] <= 6000)
model.nobuy_H = pyo.Constraint(expr=model.buy_H['S'] == 0)

model.feed_cattle_W_M = pyo.Constraint(expr=model.plant['W'] * 2.5 - model.sell_M['W'] + model.buy_M['W'] >= 200)
model.feed_cattle_C_M = pyo.Constraint(expr=model.plant['C'] * 3 - model.sell_M['C'] + model.buy_M['C'] >= 240)
model.sell_S_extra_M = pyo.Constraint(expr=model.plant['S'] * 20 >= model.sell_M['S'] + model.sell_extra_M)
model.sell_S_M = pyo.Constraint(expr=model.sell_M['S'] <= 6000)
model.nobuy_M = pyo.Constraint(expr=model.buy_M['S'] == 0)

model.feed_cattle_W_L = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
model.feed_cattle_C_L = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
model.sell_S_extra_L = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

def first_stage_profit(model):
    return -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260

model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

def second_stage_profit(model):
    total_H = -model.buy_H['W'] * 238 * model.pricefactor_H - model.buy_H['C'] * 210 * model.pricefactor_H + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 * model.pricefactor_H + model.sell_H['C'] * 150 * model.pricefactor_H
    total_M = -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150
    total_L = -model.buy_L['W'] * 238 * model.pricefactor_L - model.buy_L['C'] * 210 * model.pricefactor_L + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 * model.pricefactor_L + model.sell_L['C'] * 150 * model.pricefactor_L
    return (total_H + total_M + total_L)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.first_stage_profit + model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(land allocation) $x_1 = {model.plant['W'].value:.1f}$, $x_2 = {model.plant['C'].value:.1f}$, $x_3 = {model.plant['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action high yield) $w_1 = {model.sell_H['W'].value:.1f}$, $w_2 = {model.sell_H['C'].value:.1f}$, $w_3 = {model.sell_H['S'].value:.1f}$, $w_4 = {model.sell_extra_H.value:.1f}$"))
display(Markdown(f"(recourse purchase action high yield) $y_1 = {model.buy_H['W'].value:.1f}$, $y_2 = {model.buy_H['C'].value:.1f}$, $y_3 = {model.buy_H['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action medium yield) $w_1 = {model.sell_M['W'].value:.1f}$, $w_2 = {model.sell_M['C'].value:.1f}$, $w_3 = {model.sell_M['S'].value:.1f}$, $w_4 = {model.sell_extra_M.value:.1f}$"))
display(Markdown(f"(recourse purchase action medium yield) $y_1 = {model.buy_M['W'].value:.1f}$, $y_2 = {model.buy_M['C'].value:.1f}$, $y_3 = {model.buy_M['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action low yield) $w_1 = {model.sell_L['W'].value:.1f}$, $w_2 = {model.sell_L['C'].value:.1f}$, $w_3 = {model.sell_L['S'].value:.1f}$, $w_4 = {model.sell_extra_L.value:.1f}$"))
display(Markdown(f"(recourse purchase action low yield) $y_1 = {model.buy_L['W'].value:.1f}$, $y_2 = {model.buy_L['C'].value:.1f}$, $y_3 = {model.buy_L['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.total_expected_profit():.0f}$€"))


# ## (c) Discrete First-Stage Decisions
# 
# Consider again the case where prices are fixed and scenario-independent. The farmer possesses four fields of sizes $185$, $145$, $105$, and $65$ acres, respectively. (observe that the total of 500 acres is unchanged). For reasons of efficiency the farmer wants to raise only one type of crop on each fields. 
# 
# Formulate this model as a two-stage stochastic program with a first-stage program with binary variables and solve it using once more the extensive form of the same stochastic program.

# In[226]:


field_size = {1: 185, 2: 145, 3: 105, 4: 65}


# In[246]:


m = pyo.ConcreteModel()

# mutable parameter
m.total_acres = pyo.Param(initialize=500, mutable=True)

# sets
m.crops = pyo.Set(initialize=planting_cost.keys())
m.scenarios = pyo.Set(initialize=yield_factor.keys())
m.fields = pyo.Set(initialize=field_size.keys())

# constant parameter values
m.planting_cost = pyo.Param(m.crops, initialize=planting_cost)
m.crop_yield = pyo.Param(m.crops, initialize=crop_yield)
m.sell_price = pyo.Param(m.crops, initialize=sell_price)
m.buy_price = pyo.Param(m.crops, initialize=buy_price)
m.demand = pyo.Param(m.crops, initialize=demand)
m.cattle_feed = pyo.Param(m.crops, initialize=cattle_feed)
m.field_size = pyo.Param(m.fields, initialize=field_size)

# scenario dependent parameter values
m.yield_factor = pyo.Param(m.scenarios, initialize=yield_factor)
@m.Param(m.crops, m.scenarios)
def price_factor(m, c, s):
    return 1.0

# first stage variables

m.plant = pyo.Var(m.crops, domain=pyo.NonNegativeReals)
m.assign_field_to_crop = pyo.Var(m.fields, m.crops, domain=pyo.Binary)

# first stage constraint
@m.Constraint(m.crops)
def plant_fields(m, c):
    return m.plant[c] == sum(m.assign_field_to_crop[f, c] * m.field_size[f] for f in m.fields)

# assign only on crop to a field
@m.Constraint(m.fields)
def assignment(m, f):
    return sum(m.assign_field_to_crop[f, c] for c in m.crops) <= 1

# first stage profit
@m.Expression()
def first_stage_profit(m, c):
    return -sum(m.plant[c] * m.planting_cost[c] for c in m.crops)

# second stage variables
m.produced = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.buy = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.sell = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)

# second stage constraints
@m.Constraint(m.crops, m.scenarios)
def crops_produced(m, c, s):
    return m.produced[c, s] == m.plant[c] * m.crop_yield[c] * m.yield_factor[s]

@m.Constraint(m.crops, m.scenarios)
def demand_limit(m, c, s):
    return m.sell[c, s] <= m.demand[c]

@m.Constraint(m.crops, m.scenarios)
def balance(m, c, s):
    return m.produced[c, s] + m.buy[c, s] == m.cattle_feed[c] + m.sell[c, s]

# second stage profit
@m.Expression(m.scenarios)
def second_stage_profit(m, s):
    revenue = sum(m.sell[c, s] * m.sell_price[c] * m.price_factor[c, s] for c in m.crops) 
    expense = sum(m.buy[c, s] * m.buy_price[c] * m.price_factor[c, s] for c in m.crops)
    return revenue - expense

# Objective
@m.Objective(sense=pyo.maximize)
def total_profit(m):
    return m.first_stage_profit + sum(m.second_stage_profit[s] for s in m.scenarios) / len(m.scenarios)

pyo.SolverFactory('cbc').solve(m)
    
print(f"\nFirst Stage Profit: {m.first_stage_profit.expr():0.2f} euros")
print("\tCrop       Planted")
print("\t           (acres)")
for c in m.crops:
    print(f"\t{c:8s}   {m.plant[c].value:7.1f}")
    
print(f"\n\tField        Acres     Crop")
for f in m.fields:
    print(f"\t{f}            {field_size[f]:5.1f}", end="")
    for c in m.crops:
        if m.assign_field_to_crop[f, c]():
            print(f"        {c}")
    
print(f"\nSecond Stage / Recourse Solutions")
for s in m.scenarios:
    print(f"\nScenario {s} Profit: {m.second_stage_profit[s].expr():0.2f} euros")
    print("\tCrop      Produced       Buy      Feed      Sell")
    print("\t            (tons)     (tons)    (tons)   (tons)")
    for c in m.crops:
        sout = f"\t{c:8s}"
        sout += f"   {m.produced[c, s].value:7.1f}"
        sout += f"   {m.buy[c, s].value:7.1f}"
        sout += f"   {cattle_feed[c]:7.1f}"
        sout += f"   {m.sell[c, s].value:7.1f}"
        print(sout)

print(f"\nExpected profit = {m.total_profit():0.2f} euros")


# In[6]:


model = pyo.ConcreteModel()

model.crops = pyo.Set(initialize=['W', 'C', 'S'])
model.fields = pyo.Set(initialize=['1','2','3','4'])
model.fieldsize = pyo.Param(model.fields, initialize={'1': 185.0, '2': 145.0, '3': 105.0, '4': 65.0}, within=pyo.NonNegativeReals)

model.factor_H = 1.2 # to obtain the yields in the good weather (high yield) case by multiplying the average ones
model.factor_L = 0.8 # to obtain the yields in the bad weather (low yield) case by multiplying the average ones
model.pricefactor_H = 1.0 # to obtain the prices in the good weather (high yield) case by multiplying the average ones
model.pricefactor_L = 1.0 # to obtain the prices in the bad weather (low yield) case by multiplying the average ones

# first stage variables, which are now binary variables
model.plant_W = pyo.Var(model.fields, within=pyo.Binary) 
model.plant_C = pyo.Var(model.fields, within=pyo.Binary) 
model.plant_S = pyo.Var(model.fields, within=pyo.Binary)

# first stage constraint
model.field1 = pyo.Constraint(expr=model.plant_W['1'] + model.plant_C['1'] + model.plant_S['1'] <= 1)
model.field2 = pyo.Constraint(expr=model.plant_W['2'] + model.plant_C['2'] + model.plant_S['2'] <= 1)
model.field3 = pyo.Constraint(expr=model.plant_W['3'] + model.plant_C['3'] + model.plant_S['3'] <= 1)
model.field4 = pyo.Constraint(expr=model.plant_W['4'] + model.plant_C['4'] + model.plant_S['4'] <= 1)

# we recalculate the total surface per type of crop
model.totalWsurface = pyo.Expression(expr=np.sum([model.plant_W[i] * model.fieldsize[i] for i in model.fields]))
model.totalCsurface = pyo.Expression(expr=np.sum([model.plant_C[i] * model.fieldsize[i] for i in model.fields]))
model.totalSsurface = pyo.Expression(expr=np.sum([model.plant_S[i] * model.fieldsize[i] for i in model.fields]))

model.scenarios = pyo.Set(initialize=['H', 'M', 'L'])  # high, medium, and low yield scenarios

# second stage variables (labelled as H,M,L depending on the scenario)
# the sell_extra variables refer to the amount of beets to be sold beyond the 6000 threshold, if any

model.sell_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_H = pyo.Var(within=pyo.NonNegativeReals) 

model.sell_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_M = pyo.Var(within=pyo.NonNegativeReals)

model.sell_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_L = pyo.Var(within=pyo.NonNegativeReals)

# second stage constraints
model.feed_cattle_W_H = pyo.Constraint(expr=model.totalWsurface * 2.5 * model.factor_H - model.sell_H['W'] + model.buy_H['W'] >= 200)
model.feed_cattle_C_H = pyo.Constraint(expr=model.totalCsurface * 3 * model.factor_H - model.sell_H['C'] + model.buy_H['C'] >= 240)
model.sell_S_extra_H = pyo.Constraint(expr=model.totalSsurface * 20 * model.factor_H >= model.sell_H['S'] + model.sell_extra_H)
model.sell_S_H = pyo.Constraint(expr=model.sell_H['S'] <= 6000)
model.nobuy_H = pyo.Constraint(expr=model.buy_H['S'] == 0)

model.feed_cattle_W_M = pyo.Constraint(expr=model.totalWsurface * 2.5 - model.sell_M['W'] + model.buy_M['W'] >= 200)
model.feed_cattle_C_M = pyo.Constraint(expr=model.totalCsurface * 3 - model.sell_M['C'] + model.buy_M['C'] >= 240)
model.sell_S_extra_M = pyo.Constraint(expr=model.totalSsurface * 20 >= model.sell_M['S'] + model.sell_extra_M)
model.sell_S_M = pyo.Constraint(expr=model.sell_M['S'] <= 6000)
model.nobuy_M = pyo.Constraint(expr=model.buy_M['S'] == 0)

model.feed_cattle_W_L = pyo.Constraint(expr=model.totalWsurface * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
model.feed_cattle_C_L = pyo.Constraint(expr=model.totalCsurface * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
model.sell_S_extra_L = pyo.Constraint(expr=model.totalSsurface * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

def first_stage_profit(model):
    return -model.totalWsurface * 150 - model.totalCsurface * 230 - model.totalSsurface * 260

model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

def second_stage_profit(model):
    total_H = -model.buy_H['W'] * 238 * model.pricefactor_H - model.buy_H['C'] * 210 * model.pricefactor_H + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 * model.pricefactor_H + model.sell_H['C'] * 150 * model.pricefactor_H
    total_M = -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150
    total_L = -model.buy_L['W'] * 238 * model.pricefactor_L - model.buy_L['C'] * 210 * model.pricefactor_L + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 * model.pricefactor_L + model.sell_L['C'] * 150 * model.pricefactor_L
    return (total_H + total_M + total_L)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.first_stage_profit + model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(wheat field allocation) $[{model.plant_W['1'].value:.0f}, {model.plant_W['2'].value:.0f}, {model.plant_W['3'].value:.0f}, {model.plant_W['4'].value:.0f}]$"))
display(Markdown(f"(corn field allocation) $[{model.plant_C['1'].value:.0f}, {model.plant_C['2'].value:.0f}, {model.plant_C['3'].value:.0f}, {model.plant_C['4'].value:.0f}]$"))
display(Markdown(f"(beet field allocation) $[{model.plant_S['1'].value:.0f}, {model.plant_S['2'].value:.0f}, {model.plant_S['3'].value:.0f}, {model.plant_S['4'].value:.0f}]$"))
display(Markdown(f"(land field allocation) $x_1 = {model.totalWsurface.expr():.1f}$, $x_2 = {model.totalCsurface.expr():.1f}$, $x_3 = {model.totalSsurface.expr():.1f}$"))
display(Markdown(f"(recourse sell action high yield) $w_1 = {model.sell_H['W'].value:.1f}$, $w_2 = {model.sell_H['C'].value:.1f}$, $w_3 = {model.sell_H['S'].value:.1f}$, $w_4 = {model.sell_extra_H.value:.1f}$"))
display(Markdown(f"(recourse purchase action high yield) $y_1 = {model.buy_H['W'].value:.1f}$, $y_2 = {model.buy_H['C'].value:.1f}$, $y_3 = {model.buy_H['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action medium yield) $w_1 = {model.sell_M['W'].value:.1f}$, $w_2 = {model.sell_M['C'].value:.1f}$, $w_3 = {model.sell_M['S'].value:.1f}$, $w_4 = {model.sell_extra_M.value:.1f}$"))
display(Markdown(f"(recourse purchase action medium yield) $y_1 = {model.buy_M['W'].value:.1f}$, $y_2 = {model.buy_M['C'].value:.1f}$, $y_3 = {model.buy_M['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action low yield) $w_1 = {model.sell_L['W'].value:.1f}$, $w_2 = {model.sell_L['C'].value:.1f}$, $w_3 = {model.sell_L['S'].value:.1f}$, $w_4 = {model.sell_extra_L.value:.1f}$"))
display(Markdown(f"(recourse purchase action low yield) $y_1 = {model.buy_L['W'].value:.1f}$, $y_2 = {model.buy_L['C'].value:.1f}$, $y_3 = {model.buy_L['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.total_expected_profit():.0f}$€"))


# ## (d) Risk Averse Farmer
# 
# Consider again the setting described in (a). The farmer would normally act as a risk-averse person and simply plan for the worst case. More precisely, the farmer maximizes her profit under the worst scenario, that is the bad weather one.
# 
# (d) Solve the deterministic LP for the bad weather scenario to find the corresponding worst-case optimal land allocation. Using this land allocation, compute the loss in expected profit if that solution is taken.

# In[249]:


m = pyo.ConcreteModel()

# mutable parameter
m.total_acres = pyo.Param(initialize=500, mutable=True)

# sets
m.crops = pyo.Set(initialize=planting_cost.keys())
m.scenarios = pyo.Set(initialize=yield_factor.keys())

# constant parameter values
m.planting_cost = pyo.Param(m.crops, initialize=planting_cost)
m.crop_yield = pyo.Param(m.crops, initialize=crop_yield)
m.sell_price = pyo.Param(m.crops, initialize=sell_price)
m.buy_price = pyo.Param(m.crops, initialize=buy_price)
m.demand = pyo.Param(m.crops, initialize=demand)
m.cattle_feed = pyo.Param(m.crops, initialize=cattle_feed)

# scenario dependent parameter values
m.yield_factor = pyo.Param(m.scenarios, initialize=yield_factor)

# first stage variables

m.plant = pyo.Var(m.crops, domain=pyo.NonNegativeReals)

# first stage constraint
@m.Constraint()
def acres(m):
    return sum(m.plant[c] for c in m.crops) <= m.total_acres

# first stage profit
@m.Expression()
def first_stage_profit(m, c):
    return -sum(m.plant[c] * m.planting_cost[c] for c in m.crops)

# second stage variables
m.produced = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.buy = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)
m.sell = pyo.Var(m.crops, m.scenarios, domain=pyo.NonNegativeReals)

# second stage constraints
@m.Constraint(m.crops, m.scenarios)
def crops_produced(m, c, s):
    return m.produced[c, s] == m.plant[c] * m.crop_yield[c] * m.yield_factor[s]

@m.Constraint(m.crops, m.scenarios)
def demand_limit(m, c, s):
    return m.sell[c, s] <= m.demand[c]

@m.Constraint(m.crops, m.scenarios)
def balance(m, c, s):
    return m.produced[c, s] + m.buy[c, s] == m.cattle_feed[c] + m.sell[c, s]

# second stage profit
@m.Expression(m.scenarios)
def second_stage_profit(m, s):
    revenue = sum(m.sell[c, s] * m.sell_price[c] for c in m.crops) 
    expense = sum(m.buy[c, s] * m.buy_price[c] for c in m.crops)
    return revenue - expense

# Objective
m.worst_case_profit = pyo.Var(domain=pyo.NonNegativeReals)

@m.Constraint(m.scenarios)
def worst_case(m, s):
    return m.worst_case_profit <= m.first_stage_profit + m.second_stage_profit[s] 

@m.Objective(sense=pyo.maximize)
def total_profit(m):
    return m.worst_case_profit

pyo.SolverFactory('cbc').solve(m)
    
print(f"\nFirst Stage Profit: {m.first_stage_profit.expr():0.2f} euros")
print("\tCrop       Planted")
print("\t           (acres)")
for c in m.crops:
    print(f"\t{c:8s}   {m.plant[c].value:7.1f}")
    
print(f"\nSecond Stage / Recourse Solutions")
for s in m.scenarios:
    print(f"\nScenario {s} Profit: {m.second_stage_profit[s].expr():0.2f} euros")
    print("\tCrop      Produced       Buy      Feed      Sell")
    print("\t            (tons)     (tons)    (tons)   (tons)")
    for c in m.crops:
        sout = f"\t{c:8s}"
        sout += f"   {m.produced[c, s].value:7.1f}"
        sout += f"   {m.buy[c, s].value:7.1f}"
        sout += f"   {cattle_feed[c]:7.1f}"
        sout += f"   {m.sell[c, s].value:7.1f}"
        print(sout)

print(f"\nWorst Case Profit = {m.total_profit():0.2f} euros")


# In[7]:


# Bad weather only
model = pyo.ConcreteModel()

model.crops = pyo.Set(initialize=['W', 'C', 'S'])
model.totalacres = 500
model.factor_H = 1.2 # to obtain the yields in the good weather (high yield) case by multiplying the average ones
model.factor_L = 0.8 # to obtain the yields in the bad weather (low yield) case by multiplying the average ones

# first stage variables
model.plant = pyo.Var(model.crops, within=pyo.NonNegativeReals) 

# first stage constraint
model.total_acres = pyo.Constraint(expr=pyo.summation(model.plant) <= model.totalacres)

# second stage variables
model.sell_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_L = pyo.Var(within=pyo.NonNegativeReals)

# second stage constraints
model.feed_cattle_W_L = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
model.feed_cattle_C_L = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
model.sell_S_extra_L = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

def first_stage_profit(model):
    return -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260

model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

def second_stage_profit(model):
    return -model.buy_L['W'] * 238 - model.buy_L['C'] * 210 + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 + model.sell_L['C'] * 150

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.first_stage_profit + model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(land allocation) $x_1 = {model.plant['W'].value:.1f}$, $x_2 = {model.plant['C'].value:.1f}$, $x_3 = {model.plant['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action low yield) $w_1 = {model.sell_L['W'].value:.1f}$, $w_2 = {model.sell_L['C'].value:.1f}$, $w_3 = {model.sell_L['S'].value:.1f}$, $w_4 = {model.sell_extra_L.value:.1f}$"))
display(Markdown(f"(recourse purchase action low yield) $y_1 = {model.buy_L['W'].value:.1f}$, $y_2 = {model.buy_L['C'].value:.1f}$, $y_3 = {model.buy_L['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.total_expected_profit():.0f}$€"))


# In[8]:


# Expected profit when optimizing only against bad weather 
model = pyo.ConcreteModel()

model.crops = pyo.Set(initialize=['W', 'C', 'S'])
model.factor_H = 1.2 # to obtain the yields in the good weather (high yield) case by multiplying the average ones
model.factor_L = 0.8 # to obtain the yields in the bad weather (low yield) case by multiplying the average ones

# first stage variables are now parameters, since we set them equal to the optimal allocation for bad weather calculated in the previous cell
model.plant = pyo.Param(model.crops, within=pyo.NonNegativeReals, initialize={'W': 100, 'C': 25, 'S': 375}) 

model.scenarios = pyo.Set(initialize=['H', 'M', 'L'])  # high, medium, and low yield scenarios

# second stage variables (labelled as H,M,L depending on the scenario)
# the sell_extra variables refer to the amount of beets to be sold beyond the 6000 threshold, if any
model.sell_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_H = pyo.Var(within=pyo.NonNegativeReals) 

model.sell_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_M = pyo.Var(within=pyo.NonNegativeReals)

model.sell_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.buy_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
model.sell_extra_L = pyo.Var(within=pyo.NonNegativeReals)

# second stage constraints
model.feed_cattle_W_H = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_H - model.sell_H['W'] + model.buy_H['W'] >= 200)
model.feed_cattle_C_H = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_H - model.sell_H['C'] + model.buy_H['C'] >= 240)
model.sell_S_extra_H = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_H >= model.sell_H['S'] + model.sell_extra_H)
model.sell_S_H = pyo.Constraint(expr=model.sell_H['S'] <= 6000)
model.nobuy_H = pyo.Constraint(expr=model.buy_H['S'] == 0)

model.feed_cattle_W_M = pyo.Constraint(expr=model.plant['W'] * 2.5 - model.sell_M['W'] + model.buy_M['W'] >= 200)
model.feed_cattle_C_M = pyo.Constraint(expr=model.plant['C'] * 3 - model.sell_M['C'] + model.buy_M['C'] >= 240)
model.sell_S_extra_M = pyo.Constraint(expr=model.plant['S'] * 20 >= model.sell_M['S'] + model.sell_extra_M)
model.sell_S_M = pyo.Constraint(expr=model.sell_M['S'] <= 6000)
model.nobuy_M = pyo.Constraint(expr=model.buy_M['S'] == 0)

model.feed_cattle_W_L = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
model.feed_cattle_C_L = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
model.sell_S_extra_L = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

def first_stage_profit(model):
    return -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260

model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

def second_stage_profit(model):
    total_H = -model.buy_H['W'] * 238 - model.buy_H['C'] * 210 + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 + model.sell_H['C'] * 150
    total_M = -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150
    total_L = -model.buy_L['W'] * 238 - model.buy_L['C'] * 210 + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 + model.sell_L['C'] * 150
    return (total_H + total_M + total_L)/3.0

model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

def total_profit(model):
    return model.first_stage_profit + model.second_stage_profit

model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

result = cbc_solver.solve(model)
display(Markdown(f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"))
display(Markdown(f"**Solution:**"))
display(Markdown(f"(land allocation) $x_1 = {model.plant['W']:.1f}$, $x_2 = {model.plant['C']:.1f}$, $x_3 = {model.plant['S']:.1f}$"))
display(Markdown(f"(recourse sell action high yield) $w_1 = {model.sell_H['W'].value:.1f}$, $w_2 = {model.sell_H['C'].value:.1f}$, $w_3 = {model.sell_H['S'].value:.1f}$, $w_4 = {model.sell_extra_H.value:.1f}$"))
display(Markdown(f"(recourse purchase action high yield) $y_1 = {model.buy_H['W'].value:.1f}$, $y_2 = {model.buy_H['C'].value:.1f}$, $y_3 = {model.buy_H['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action medium yield) $w_1 = {model.sell_M['W'].value:.1f}$, $w_2 = {model.sell_M['C'].value:.1f}$, $w_3 = {model.sell_M['S'].value:.1f}$, $w_4 = {model.sell_extra_M.value:.1f}$"))
display(Markdown(f"(recourse purchase action medium yield) $y_1 = {model.buy_M['W'].value:.1f}$, $y_2 = {model.buy_M['C'].value:.1f}$, $y_3 = {model.buy_M['S'].value:.1f}$"))
display(Markdown(f"(recourse sell action low yield) $w_1 = {model.sell_L['W'].value:.1f}$, $w_2 = {model.sell_L['C'].value:.1f}$, $w_3 = {model.sell_L['S'].value:.1f}$, $w_4 = {model.sell_extra_L.value:.1f}$"))
display(Markdown(f"(recourse purchase action low yield) $y_1 = {model.buy_L['W'].value:.1f}$, $y_2 = {model.buy_L['C'].value:.1f}$, $y_3 = {model.buy_L['S'].value:.1f}$"))
display(Markdown(f"**Maximizes objective value to:** ${model.total_expected_profit():.0f}$€"))


# (e) A different approach situation be to require a reasonable minimum profit under the worst case. Find the solution that maximizes the expected profit under the constraint that in the worst case the profit does not fall below $58.000$ euro. What is now the loss in expected profit?
# 
# Repeat the same optimization also with other values of minimal profit: $56.000$, $54.000$, $52.000$, $50.000$, and $48.000$ euro. Graph the curve of expected profit loss and compare the associated optimal decisions.

# In[9]:


expectedprofit_noconstraints = 108390

def FarmersWithMinimumProfit(minprofit):
    model = pyo.ConcreteModel()

    model.crops = pyo.Set(initialize=['W', 'C', 'S'])
    model.totalacres = 500
    model.factor_H = 1.2 # to obtain the yields in the good weather (high yield) case by multiplying the average ones
    model.factor_L = 0.8 # to obtain the yields in the bad weather (low yield) case by multiplying the average ones
    model.pricefactor_H = 1.0 # to obtain the prices in the good weather (high yield) case by multiplying the average ones
    model.pricefactor_L = 1.0 # to obtain the prices in the bad weather (low yield) case by multiplying the average ones

    # first stage variables
    model.plant = pyo.Var(model.crops, within=pyo.NonNegativeReals) 

    # first stage constraint
    model.total_acres = pyo.Constraint(expr=pyo.summation(model.plant) <= model.totalacres)

    model.scenarios = pyo.Set(initialize=['H', 'M', 'L'])  # high, medium, and low yield scenarios

    # second stage variables (labelled as H,M,L depending on the scenario)
    # the sell_extra variables refer to the amount of beets to be sold beyond the 6000 threshold, if any
    model.sell_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
    model.buy_H = pyo.Var(model.crops, within=pyo.NonNegativeReals)
    model.sell_extra_H = pyo.Var(within=pyo.NonNegativeReals) 

    model.sell_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
    model.buy_M = pyo.Var(model.crops, within=pyo.NonNegativeReals)
    model.sell_extra_M = pyo.Var(within=pyo.NonNegativeReals)

    model.sell_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
    model.buy_L = pyo.Var(model.crops, within=pyo.NonNegativeReals)
    model.sell_extra_L = pyo.Var(within=pyo.NonNegativeReals)

    # second stage constraints
    model.feed_cattle_W_H = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_H - model.sell_H['W'] + model.buy_H['W'] >= 200)
    model.feed_cattle_C_H = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_H - model.sell_H['C'] + model.buy_H['C'] >= 240)
    model.sell_S_extra_H = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_H >= model.sell_H['S'] + model.sell_extra_H)
    model.sell_S_H = pyo.Constraint(expr=model.sell_H['S'] <= 6000)
    model.nobuy_H = pyo.Constraint(expr=model.buy_H['S'] == 0)

    model.feed_cattle_W_M = pyo.Constraint(expr=model.plant['W'] * 2.5 - model.sell_M['W'] + model.buy_M['W'] >= 200)
    model.feed_cattle_C_M = pyo.Constraint(expr=model.plant['C'] * 3 - model.sell_M['C'] + model.buy_M['C'] >= 240)
    model.sell_S_extra_M = pyo.Constraint(expr=model.plant['S'] * 20 >= model.sell_M['S'] + model.sell_extra_M)
    model.sell_S_M = pyo.Constraint(expr=model.sell_M['S'] <= 6000)
    model.nobuy_M = pyo.Constraint(expr=model.buy_M['S'] == 0)

    model.feed_cattle_W_L = pyo.Constraint(expr=model.plant['W'] * 2.5 * model.factor_L - model.sell_L['W'] + model.buy_L['W'] >= 200)
    model.feed_cattle_C_L = pyo.Constraint(expr=model.plant['C'] * 3 * model.factor_L - model.sell_L['C'] + model.buy_L['C'] >= 240)
    model.sell_S_extra_L = pyo.Constraint(expr=model.plant['S'] * 20 * model.factor_L >= model.sell_L['S'] + model.sell_extra_L)
    model.sell_S_L = pyo.Constraint(expr=model.sell_L['S'] <= 6000)
    model.nobuy_L = pyo.Constraint(expr=model.buy_L['S'] == 0)

    def first_stage_profit(model):
        return -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260

    model.first_stage_profit = pyo.Expression(rule=first_stage_profit)

    def second_stage_profit(model):
        total_H = -model.buy_H['W'] * 238 * model.pricefactor_H - model.buy_H['C'] * 210 * model.pricefactor_H + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 * model.pricefactor_H + model.sell_H['C'] * 150 * model.pricefactor_H
        total_M = -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150
        total_L = -model.buy_L['W'] * 238 * model.pricefactor_L - model.buy_L['C'] * 210 * model.pricefactor_L + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 * model.pricefactor_L + model.sell_L['C'] * 150 * model.pricefactor_L
        return (total_H + total_M + total_L)/3.0

    model.second_stage_profit = pyo.Expression(rule=second_stage_profit)

    def total_profit(model):
        return model.first_stage_profit + model.second_stage_profit

    model.total_expected_profit = pyo.Objective(rule=total_profit, sense=pyo.maximize)

    model.totalprofit_H = pyo.Expression(expr = -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260 -model.buy_H['W'] * 238 * model.pricefactor_H - model.buy_H['C'] * 210 * model.pricefactor_H + 36 * model.sell_H['S'] + 10 * model.sell_extra_H + model.sell_H['W'] * 170 * model.pricefactor_H + model.sell_H['C'] * 150 * model.pricefactor_H)
    model.totalprofit_M = pyo.Expression(expr = -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260 -model.buy_M['W'] * 238 - model.buy_M['C'] * 210 + 36 * model.sell_M['S'] + 10 * model.sell_extra_M + model.sell_M['W'] * 170 + model.sell_M['C'] * 150)
    model.totalprofit_L = pyo.Expression(expr = -model.plant["W"] * 150 - model.plant["C"] * 230 - model.plant["S"] * 260 -model.buy_L['W'] * 238 * model.pricefactor_L - model.buy_L['C'] * 210 * model.pricefactor_L + 36 * model.sell_L['S'] + 10 * model.sell_extra_L + model.sell_L['W'] * 170 * model.pricefactor_L + model.sell_L['C'] * 150 * model.pricefactor_L)
    model.minimum_profit = pyo.Constraint(expr=model.totalprofit_L >= minprofit)
    result = cbc_solver.solve(model)

    display(Markdown(f"**Minimum profit threshold** of ${minprofit:.0f}$€ leads to an **optimal expected profit** of ${model.total_expected_profit():.0f}$€, with a **loss** of ${expectedprofit_noconstraints - model.total_expected_profit():.0f}$€"))
    return model.total_expected_profit()

profitthresholds = [48000,50000,52000,54000,56000,58000]
for minprofit in profitthresholds:
    FarmersWithMinimumProfit(minprofit)

