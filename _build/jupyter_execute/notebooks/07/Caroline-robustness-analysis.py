#!/usr/bin/env python
# coding: utf-8

# # Caroline on robust steroids
# 
# As we know, BIM produces logic and memory chips using copper, silicon, germanium and plastic. Each chip has the following consumption of materials:
# 
# | chip   | copper | silicon | germanium | plastic |
# |:-------|-------:|--------:|----------:|--------:|
# |Logic   |    0.4 |       1 |           |       1 |
# |Memory  |    0.2 |         |         1 |       1 |
# 
# BIM hired Caroline to manage the acquisition and the inventory of these raw materials. Caroline conducted a data analysis which lead to the following prediction of monthly demands: 
# 
# | chip   | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
# |:-------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
# |Logic   |  88 | 125 | 260 | 217 | 238 | 286 | 248 | 238 | 265 | 293 | 259 | 244 |
# |Memory  |  47 |  62 |  81 |  65 |  95 | 118 |  86 |  89 |  82 |  82 |  84 | 66  |
# 
# As you recall, BIM has the following stock at the moment:
# 
# |copper|silicon|germanium|plastic|
# |-----:|------:|--------:|------:|
# |   480|  1000 |     1500|  1750 |
# 
# BIM would like to have at least the following stock at the end of the year:
# 
# |copper|silicon|germanium|plastic|
# |-----:|------:|--------:|------:|
# |   200|   500 |      500|  1000 |
# 
# Each product can be acquired at each month, but the unit prices vary as follows:
# 
# | product  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
# |:---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
# |copper    |   1 |   1 |   1 |   2 |   2 |   3 |   3 |   2 |   2 |   1 |   1 |   2 |
# |silicon   |   4 |   3 |   3 |   3 |   5 |   5 |   6 |   5 |   4 |   3 |   3 |   5 |
# |germanium |   5 |   5 |   5 |   3 |   3 |   3 |   3 |   2 |   3 |   4 |   5 |   6 |
# |plastic   | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
# 
# The inventory is limited by a capacity of a total of 9000 units per month, regardless of the composition of products in stock. 
# The holding costs of the inventory are 0.05 per unit per month regardless of the product.
# 
# Caroline cannot spend more than 5000 per month on acquisition.
# 
# Note that Caroline aims at minimizing the acquisition and holding costs of the materials while meeting the required quantities for production. The production is made to order, meaning that no inventory of chips is kept.
# 
# Please model and solve this material planning problem with the data above. 

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# In[2]:


from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

demand_data = '''
chip, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
Logic, 88, 125, 260, 217, 238, 286, 248, 238, 265, 293, 259, 244
Memory, 47, 62, 81, 65, 95, 118, 86, 89, 82, 82, 84, 66
'''

demand_chips = pd.read_csv(StringIO(demand_data), index_col='chip')
display(demand_chips)

price_data = '''
product, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
copper, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 2
silicon, 4, 3, 3, 3, 5, 5, 6, 5, 4, 3, 3, 5
germanium, 5, 5, 5, 3, 3, 3, 3, 2, 3, 4, 5, 6
plastic, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
'''

price = pd.read_csv(StringIO(price_data), index_col='product')
display(price)


# In[3]:


use = dict()
use['Logic'] = {'silicon': 1, 'plastic': 1, 'copper': 4}
use['Memory'] = {'germanium': 1, 'plastic': 1, 'copper': 2}
use = pd.DataFrame.from_dict(use).fillna(0).astype(int)
material_demand = use.dot(demand_chips)

existing = pd.Series({'silicon': 1000, 'germanium': 1500, 'plastic': 1750, 'copper': 4800 })
eot_inventory = pd.Series({'silicon':  500, 'germanium':  500, 'plastic': 1000, 'copper': 2000 })

# we store all the problemd data in one object to easily perform folding-horizon simulations
def initialize_problem_data():
    problem_data = {
        "price": price.copy(deep=True),
        "inventory_cost": 0.05,
        "material_demand": material_demand.copy(deep=True),
        "demand_chips_ref": demand_chips.copy(deep=True),
        "demand_chips_simulation": demand_chips.copy(deep=True),
        "use": use.copy(deep=True),
        "existing": existing.copy(deep=True),
        "eot_inventory": eot_inventory.copy(deep=True),
        "stock_limit": 9000,
        "month_budget": 2500
    }
    return problem_data

problem_data = initialize_problem_data()


# In[4]:


# This is the old function

def BIMProductAcquisitionAndInventory(problem_data):
    demand = problem_data["use"].dot( problem_data["demand_chips_ref"] )
    acquisition_price = problem_data["price"]
    existing = problem_data["existing"]
    desired = problem_data["eot_inventory"]
    stock_limit = problem_data["stock_limit"]
    month_budget = problem_data["month_budget"]
    
    m = pyo.ConcreteModel( 'Product acquisition and inventory' )
    
    periods  = demand.columns
    products = demand.index 
    first    = periods[0] 
    prev     = { j : i for i,j in zip(periods,periods[1:]) }
    last     = periods[-1]
    
    m.T = pyo.Set( initialize=periods )
    m.P = pyo.Set( initialize=products )
    
    m.PT = m.P * m.T # to avoid internal set bloat
    
    m.x = pyo.Var(m.PT, within=pyo.NonNegativeReals)
    m.s = pyo.Var(m.PT, within=pyo.NonNegativeReals)
    
    @m.Param(m.PT)
    def pi(m,p,t):
        return acquisition_price.loc[p][t]
    
    @m.Param(m.PT)
    def h(m,p,t): 
        return .05 # the holding cost
    
    @m.Param(m.PT)
    def delta(m,t,p):
        return demand.loc[t,p]
    
    @m.Expression()
    def acquisition_cost( m ):
        return pyo.quicksum( m.pi[p,t] * m.x[p,t] for p in m.P for t in m.T )
    
    @m.Expression()
    def inventory_cost( m ):
        return pyo.quicksum( m.h[p,t] * m.s[p,t] for p in m.P for t in m.T )
    
    @m.Objective(sense=pyo.minimize)
    def total_cost(m):
        return m.acquisition_cost + m.inventory_cost
    
    @m.Constraint(m.PT)
    def balance(m, p, t):
        if t == first:
            return existing[p] + m.x[p,t] == m.delta[p,t] + m.s[p,t]
        else:
            return m.x[p,t] + m.s[p,prev[t]] == m.delta[p,t] + m.s[p,t]
        
    @m.Constraint(m.P)
    def finish(m, p):
        return m.s[p,last] >= desired[p]
    
    @m.Constraint(m.T)
    def inventory(m, t):
        return pyo.quicksum( m.s[p,t] for p in m.P ) <= stock_limit
    
    @m.Constraint(m.T)
    def budget(m, t):
        return pyo.quicksum( m.pi[p,t]*m.x[p,t] for p in m.P ) <= month_budget
    
    return m

m = BIMProductAcquisitionAndInventory(problem_data)
pyo.SolverFactory('mosek').solve(m)
print(m.total_cost())

problem_data["purchases"] = ShowTableOfPyomoVariables(m.x, m.P, m.T)
problem_data["stock"] = ShowTableOfPyomoVariables(m.s, m.P, m.T)

display(problem_data["purchases"])
display(problem_data["stock"])


# In[ ]:


m = BIMProductAcquisitionAndInventory(problem_data)
pyo.SolverFactory('cbc').solve(m)


# In[ ]:


################# SIMULATION OF THE REAL-TIME PERFORMANCE #######################

# Conceptual
#1) Decide on how to perturb the demand
#2) Decide what is done given the perturbed demand - which product has the most priority
#3) Decide what data to collect along the simulation

# Implementation
#1) Create a function that given demand trajectory and solution simulates by moving step by step, calling a sub-function
#2) Create a function that given initial state at time t-1 executes the "what do do" strategy so that we obtain the immediate decisions 
#3) Update the data and move to the next state

# How to represent a demand trajectory? Data frame with columns as period names
# What data to output per period: 
# (i) missed demand per each chip category, 
# (ii) inventory for each raw material (start or end?)

# Create a class to store all the data of the current solution so that it's easy to retrieve
import numpy as np
def simulation_per_trajectory(purchases, existing, demand_chips, use):
    
    # Set up the table to store inventory evolution
    inventory = pd.DataFrame(index = purchases.index, columns = purchases.columns)
    inventory = pd.concat([pd.DataFrame(existing, index = existing.index, columns = ["existing"]), inventory], axis = 1)
    
    #print(inventory)
    # Set up the DF to store missed demand information
    missed_demand = pd.DataFrame(np.zeros((len(demand_chips.index), len(purchases.columns))), index = demand_chips.index, columns = purchases.columns)
    
    # proper simulation
    for period in inventory.columns[1:]:
        minimize_missed_demand_in_period(inventory, missed_demand, purchases, existing, demand_chips, use, period)
        
    return inventory.iloc[:, 1:], missed_demand

def minimize_missed_demand_in_period(inventory, missed_demand, purchases, existing, demand_chips, use, period = None):
    m = pyo.ConcreteModel('In period')
    
    periods  = inventory.columns
    first    = periods[0] 
    prev     = { j : i for i,j in zip(periods,periods[1:]) }
    last     = periods[-1]
    
    m.P = pyo.Set( initialize=list(use.columns)  )
    m.M = pyo.Set( initialize=list(use.index) )
    # decision variable: nb of chips to produce >= 0
    m.x = pyo.Var( m.P, within=pyo.NonNegativeReals )
    # decision variable: missed demand
    m.s = pyo.Var( m.P, within=pyo.NonNegativeReals )
    
    # constraint: per resource we don't use more than there is
    @m.Constraint(m.M)
    def resource_constraint(m, i):
        return pyo.quicksum(m.x[p] * use.loc[i, p] for p in m.P) <= inventory.loc[i, prev[period]] + purchases.loc[i, period]
    
    # constraint: production + missed demand = total demand in this period
    @m.Constraint(m.P)
    def produced_plus_unmet(m, p):
        return m.x[p] + m.s[p] == demand_chips.loc[p, period]
    
    # objective - minimize the missed demand
    @m.Objective( sense=pyo.minimize )
    def total_unmet(m):
        return pyo.quicksum(m.s[p] for p in m.P)
    
    # solve
    pyo.SolverFactory( 'gurobi' ).solve(m)
    
    # update inventory
    for i in m.M:
        inventory.loc[i, period] = inventory.loc[i, prev[period]] + purchases.loc[i, period] \
                - sum([pyo.value(m.x[p]) * use.loc[i, p] for p in m.P])
    
    # update missed demand
    for p in m.P:
        missed_demand.loc[p, period] = pyo.value(m.s[p])
    
    return 0

def simulate_performance(problem_data, n = 50, rho = 0.2):
    results = []

    for i in range(n):
        perturbed_demand = problem_data["demand_chips_simulation"].applymap(lambda x: x * (1 + rho * (1 - 2 * np.random.rand())))
        inv, md = simulation_per_trajectory(problem_data["purchases"], problem_data["existing"], perturbed_demand, use)
        results.append({"inventory": inv, "missing_demand": md})
        
    MissingDemand = pd.concat([i["missing_demand"] for i in results], keys = [i for i in range(len(results))])
    MissingDemand = MissingDemand.astype('float').swaplevel()
    
    InventoryEvolution = pd.concat([i["inventory"] for i in results], keys = [i for i in range(len(results))])
    InventoryEvolution = InventoryEvolution.astype('float').swaplevel()

    return {"MissingDemand": MissingDemand, 
            "InventoryEvolution": InventoryEvolution}

def report(MissingDemand, InventoryEvolution, problem_data, plot_name = "Nominal"):
    to_analyze = MissingDemand
    
    # list to store DFs with per-group computed quantiles at various levels
    average_missed_demand = to_analyze.groupby(level = 0).mean().transpose()

    # build a plot with as many subplots as there are chip types
    fig, axis = plt.subplots(figsize = (11, 4))
    average_missed_demand.plot(ax = axis, drawstyle='steps-mid',grid=True)#, xticks = average_missed_demand.index)
    plt.xticks(ticks = np.arange(len(average_missed_demand.index)), labels = average_missed_demand.index)
    #axis.set_xticks(ticks=average_missed_demand.index, labels=average_missed_demand.index)
    axis.set_title("Missed demand of chips under " + str(rho * 100) + "% uncertainty")

    fig.tight_layout(pad=3.0)
    plt.savefig("Missed_demand_" + plot_name + ".pdf")
    
    realized_inv_cost = InventoryEvolution.groupby(level = 0).mean().sum(axis = 1).sum() * problem_data["inventory_cost"]
    print("Purchasing cost: ", (problem_data["price"] * problem_data["purchases"]).sum().sum())
    print("Assumed inventory cost: ", (problem_data["stock"] * problem_data["inventory_cost"]).sum().sum())
    print("Simulated inventory cost:", str(realized_inv_cost))
    print("Simulated average missing demand: ", dict(to_analyze.groupby(level = 0).mean().sum(axis = 1)))


# In[ ]:


rho = 0.05
N_sim = 50

print("================= Nominal model =====================")
problem_data["demand_chips_ref"] = demand_chips
m = BIMProductAcquisitionAndInventory(problem_data)

pyo.SolverFactory('cbc').solve(m)
problem_data["purchases"] = ShowTableOfPyomoVariables( m.x, m.P, m.T )
problem_data["stock"] = ShowTableOfPyomoVariables( m.s, m.P, m.T )

SimResults = simulate_performance(problem_data,
                                  N_sim, 
                                  rho)

report(SimResults["MissingDemand"],
       SimResults["InventoryEvolution"],
       problem_data,
       "nominal")


# In[ ]:





# In[ ]:




