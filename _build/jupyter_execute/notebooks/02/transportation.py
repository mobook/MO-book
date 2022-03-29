#!/usr/bin/env python
# coding: utf-8

# # Transportation Models
# 
# This notebook presents a transportation model to optimally allocate the delivery of a commodity from multiple sources to multiple destinations. The notebook also presents techniques for Pyomo modeling and reporting including: 
# 
# * `pyo.Expression` decorator
# * Accessing the duals (i.e., shadow prices)
# * Methods for reporting the solution and duals.
#     * Pyomo `.display()` method for Pyomo objects
#     * Manually formatted reports
#     * Pandas 
#     * Graphviz for display of results as a directed graph.
#     
# The model invites a discussion of the pitfalls in optimizing a global objective for customers who may have an uneven share of the resulting benefits.

# In[25]:


import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Distributing Gasoline to Franchise Operators
# 
# YaYa Gas-n-Grub is franchisor and operator for a network of regional convenience stores selling gasoline and convenience items in the United States. Each store is individually owned by a YaYa Gas-n-Grub franchisee who pays a fixed fee to the franchisor for services.
# 
# Gasoline is delivered by truck from regional distribution terminals. Each delivery truck carrys 8,000 gallons delivered at a fixed charge of 700 dollars per delivery, or 8.75 cents per gallon. The franchise owners are eager to reduce delivery costs to boost profits.
# 
# YaYa Gas-n-Grub decides to accept proposals from other distribution terminals, "A" and "B", to supply the franchise operators. They each provide delivery pricing based on location. Since they already have existing customers, "A" and "B" can only provide a limited amount of gasoline to new customers, 100,000 and 150,000 gallons respectively. The only difference between the new suppliers and the current supplier is the delivery charge.
# 
# The following chart shows the pricing of gasoline delivery in cents/gallon.
# 
# | Store Owner | Demand |  Terminal A | Terminal B | Current Supplier |
# | :-------- | ------------: | ---------: | -------: | --------: |
# | Alice | 30,000 | 8.3 | 10.2 | 8.75 |
# | Badri  | 40,000 | 8.1 | 12.0 | 8.75 |
# | Cara  | 50,000 | 8.3 | - | 8.75 |
# | Dan   | 80,000 | 9.3 | 8.0 |  8.75 |
# | Emma  | 30,000 | 10.1 | 10.0 | 8.75 |
# | Fujita | 45,000 | 9.8 | 10.0 | 8.75 |
# | Grace | 80,000 | -  | 8.0 | 8.75 |
# | Helen | 18,000 | 7.5 | 10.0 | 8.75 |
# | **TOTALS**| 313,000 | 100,000 | 150,000 | 500, 000 | 370,000 |
# 
# ### Task
# 
# As operator of YaYa Gas-n-Grub, allocate the delivery of gasoline to minimize overall cost to the frachise owners.
# 
# ### Analysis
# 
# The following model presents a global objective to minimize the total cost of delivery to all franchise owners. But there is no guarantee that each owner will benefit equally or at all. Suppose, for example, that a majority of the franchise owners must agree with the new contract in order to proceed. Will the franchise owers agree?  Who might object to the new arrangment? Could the model be modified to assure that each operator gains at least some share of the global objective?

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt

rates = pd.DataFrame({
    "Alice": {"Terminal A": 8.3, "Terminal B": 10.2, "Current Supplier": 8.75},
    "Badri": {"Terminal A": 8.1, "Terminal B": 12.0, "Current Supplier": 8.75},
    "Cara": {"Terminal A": 8.3, "Terminal B": 100, "Current Supplier": 8.75},
    "Dan": {"Terminal A": 9.3, "Terminal B": 8.0, "Current Supplier": 8.75},
    "Emma": {"Terminal A": 10.1, "Terminal B": 10.0, "Current Supplier": 8.75},
    "Fujita": {"Terminal A": 9.8, "Terminal B": 10.0, "Current Supplier": 8.75},
    "Grace": {"Terminal A": 100, "Terminal B": 8.0, "Current Supplier": 8.75},
    "Helen": {"Terminal A": 7.5, "Terminal B": 10.0, "Current Supplier": 8.75}}).T

display(rates)

demand = pd.Series({
    "Alice": 30000,
    "Badri": 40000,
    "Cara": 50000,
    "Dan": 20000,
    "Emma": 30000,
    "Fujita": 45000,
    "Grace": 80000,
    "Helen": 18000})

supply = pd.Series({
    "Terminal A": 100000,
    "Terminal B": 80000,
    "Current Supplier": 500000})

fix, ax = plt.subplots(1, 2)
demand.plot(kind="bar", ax=ax[0], title=f"Demand = {demand.sum()}")
supply.plot(kind="bar", ax=ax[1], title=f"Supply = {supply.sum()}")
plt.tight_layout()


# In[27]:


import pyomo.environ as pyo

def transport(supply, demand, rates):
    m = pyo.ConcreteModel()

    m.SOURCES = pyo.Set(initialize=rates.columns)
    m.DESTINATIONS = pyo.Set(initialize=rates.index)

    m.x = pyo.Var(m.DESTINATIONS, m.SOURCES, domain=pyo.NonNegativeReals)

    @m.Param(m.DESTINATIONS, m.SOURCES)
    def Rates(m, dst, src):
        return rates.loc[dst, src]

    @m.Objective(sense=pyo.minimize)
    def cost(m):
        return sum(m.Rates[dst, src]*m.x[dst, src] for dst, src in m.DESTINATIONS * m.SOURCES)
    
    @m.Expression(m.DESTINATIONS)
    def cost_to(m, dst):
        return sum(m.Rates[dst, src]*m.x[dst, src] for src in m.SOURCES)

    @m.Expression(m.DESTINATIONS)
    def shipped_to(m, dst):
        return sum(m.x[dst, src] for src in m.SOURCES)

    @m.Expression(m.SOURCES)
    def shipped_from(m, src):
        return sum(m.x[dst, src] for dst in m.DESTINATIONS)

    @m.Constraint(m.SOURCES)
    def supply_constraint(m, src):
        return m.shipped_from[src] <= supply[src]

    @m.Constraint(m.DESTINATIONS)
    def demand_constraint(m, dst):
        return m.shipped_to[dst] >= demand[dst]

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory('cbc').solve(m)

    return m

m = transport(supply, demand, rates/100)

results = pd.DataFrame({dst: {src: m.x[dst, src]() for src in m.SOURCES} for dst in m.DESTINATIONS}).T
results["current costs"] = 700*demand/8000
results["contract costs"] = pd.Series({dst: m.cost_to[dst]() for dst in m.DESTINATIONS})
results["savings"] = results["current costs"].round(1) - results["contract costs"].round(1)
results["marginal cost"]  = pd.Series({dst: m.dual[m.demand_constraint[dst]] for dst in m.DESTINATIONS})

print(f"Old Delivery Costs = $ {sum(demand)*700/8000}")
print(f"New Delivery Costs = $ {m.cost()}")
display(results)


# ## Reporting Solutions
# 
# Pyomo models can produce considerable amounts of data that must be summarized and presented for analysis and decision making. In this application, for example, the individual franchise owners receive differing amounts of savings which is certain to result in considerable discussion and possibly negotiation with the franchisor. 
# 
# The following cells demonstrate techniques for extracting and displaying information generated by a Pyomo model. 

# ### Pyomo `.display()` method
# 
# Pyomo provides a default `.display()` method for most Pyomo objects. The default display is often sufficient for model reporting requirements, particularly when initially developing a new application.

# In[33]:


# display elements of sets
m.SOURCES.display()
m.DESTINATIONS.display()


# In[34]:


# display elements of an indexed parameter
m.Rates.display()


# In[35]:


# display elements of Pyomo Expression
m.shipped_to.display()


# In[36]:


m.shipped_from.display()


# In[37]:


# display Pyomo Objective
m.cost.display()


# In[38]:


# display indexed Pyomo Constraint
m.supply_constraint.display()
m.demand_constraint.display()


# In[39]:


# display Pyomo decision variables
m.x.display()


# ### Manually formatted reports
# 
# Following solution, the value associated with Pyomo objects are returned by calling the object as a function. The following cell demonstrates the construction of a custom report using Python f-strings and Pyomo methods.

# In[18]:


# Objective report
print("\nObjective: cost")
print(f"cost = {m.cost()}")

# Constraint reports
print("\nConstraint: supply_constraint")
for src in m.SOURCES:
    print(f"{src:12s}  {m.supply_constraint[src]():8.2f}  {m.dual[m.supply_constraint[src]]:8.2f}")

print("\nConstraint: demand_constraint")
for dst in m.DESTINATIONS:
    print(f"{dst:12s}  {m.demand_constraint[dst]():8.2f}  {m.dual[m.demand_constraint[dst]]:8.2f}")

# Decision variable reports
print("\nDecision variables: x")
for src in m.SOURCES:
    for dst in m.DESTINATIONS:
        print(f"{src:12s} -> {dst:12s}  {m.x[dst, src]():8.2f}")
    print()


# ### Pandas
# 
# The Python Pandas library provides a highly flexible framework for data science applications. The next cell demonstrates the translation of Pyomo object values to Pandas DataFrames

# In[19]:


suppliers = pd.DataFrame({src: {"supply": supply[src], 
                              "shipped": m.supply_constraint[src](), 
                              "sensitivity": m.dual[m.supply_constraint[src]]}
                          for src in m.SOURCES}).T

display(suppliers)

customers = pd.DataFrame({dst: {"demand": demand[dst], 
                              "shipped": m.demand_constraint[dst](), 
                              "sensitivity": m.dual[m.demand_constraint[dst]]}
                          for dst in m.DESTINATIONS}).T

display(customers)

shipments = pd.DataFrame({dst: {src: m.x[dst, src]() for src in m.SOURCES} for dst in m.DESTINATIONS}).T
display(shipments)
shipments.plot(kind="bar")


# ### Graphviz
# 
# The `graphviz` utility is a collection of tools for visually graphs and directed graphs. Unfortunately, the package can be troublesome to install on laptops in a way that is compatable with many JupyterLab installations. Accordingly, the following cell is intended for use on Google Colab which provides a preinstalled version of `graphviz`.

# In[20]:


import graphviz
from graphviz import Digraph
import sys

if "google.colab" in sys.modules:

    dot = Digraph(
        node_attr = {"fontsize": "10", "shape": "rectangle", "style": "filled"},
        edge_attr = {"fontsize": "10"}
    )

    for src in m.SOURCES:
        label = f"{src}"                 + f"\nsupply = {supply[src]}"                 + f"\nshipped = {m.supply_constraint[src]()}"                 + f"\nsens  = {m.dual[m.supply_constraint[src]]}"
        dot.node(src, label=label, fillcolor="lightblue")

    for dst in m.DESTINATIONS:
        label = f"{dst}"                 + f"\ndemand = {demand[dst]}"                + f"\nshipped = {m.demand_constraint[dst]()}"                 + f"\nsens  = {m.dual[m.demand_constraint[dst]]}"
        dot.node(dst, label=label, fillcolor="gold")

    for src in m.SOURCES:
        for dst in m.DESTINATIONS:
            if m.x[dst, src]() > 0:
                dot.edge(src, dst, f"rate = {rates.loc[dst, src]}\nshipped = {m.x[dst, src]()}")

    display(dot)


# In[15]:




