#!/usr/bin/env python
# coding: utf-8

# # Transportation Models

# In[1]:


import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Application: Distributing Gasoline to Franchise Operators
# 
# YaYa Gas-n-Grub operates a network of eight regional convenience selling gasoline and convenience items. The gasoline is replenshed from a regional distribution terminal. The stores an average of about 4,000 gallons daily for an average profit of about \\$100 for the franchise owners, but sales vary widely based on location. Gasoline is distributed by truck from the distribution terminals, each truck carrying 8,000 gallons. Under the current business agreement, the gasoline is delivered at a fixed charge of \\$700 per delivery, or 11.43 cents per gallon. The franchise owners are eager to reduce delivery costs to boost profits, but the distributor claims a lower price would be unsustainable.
# 
# YaYa Gas-n-Grub decides to accept proposals from other distribution terminals, "A" and "B" to supply the franchise operators. They each provide delivery pricing based on location. Since they already have existing customers, can only provide a limited amount of gasoline to new customers, 100,000 and 150,000 gallons respectively. The only difference between the new suppliers and the incumbant is the delivery charge, the actual cost of gasoline is determined on the pipeline market is the same among the distributors. 
# 
# The following chart shows the pricing of gasoline delivery in cents/gallon.
# 
# | Store Owner | Demand |  Termina A | Terminal B | Incumbant |
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
# A majority of the franchise owners must agree with the new contract in order to proceed. Will the franchise owers agree?  Who might object to the new arrangment?
# 

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

rates = pd.DataFrame({
    "Alice": {"Terminal A": 8.3, "Terminal B": 10.2, "Incumbant": 8.75},
    "Badri": {"Terminal A": 8.1, "Terminal B": 12.0, "Incumbant": 8.75},
    "Cara": {"Terminal A": 8.3, "Terminal B": 100, "Incumbant": 8.75},
    "Dan": {"Terminal A": 9.3, "Terminal B": 8.0, "Incumbant": 8.75},
    "Emma": {"Terminal A": 10.1, "Terminal B": 10.0, "Incumbant": 8.75},
    "Fujita": {"Terminal A": 9.8, "Terminal B": 10.0, "Incumbant": 8.75},
    "Grace": {"Terminal A": 100, "Terminal B": 8.0, "Incumbant": 8.75},
    "Helen": {"Terminal A": 7.5, "Terminal B": 10.0, "Incumbant": 8.75}}).T

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
    "Incumbant": 500000})

fix, ax = plt.subplots(1, 2)
demand.plot(kind="bar", ax=ax[0], title=f"Demand = {demand.sum()}")
supply.plot(kind="bar", ax=ax[1], title=f"Supply = {supply.sum()}")
plt.tight_layout()


# In[3]:


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


# ## Pyomo solution reports

# In[4]:


m.SOURCES.display()


# In[5]:


m.DESTINATIONS.display()


# In[6]:


m.Rates.display()


# In[7]:


m.shipped_to.display()


# In[8]:


m.shipped_from.display()


# In[9]:


m.cost.display()


# In[10]:


m.supply_constraint.display()


# In[11]:


m.demand_constraint.display()


# In[12]:


m.x.display()


# ## Manually formatted reports

# In[13]:


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


# ## Pandas

# In[14]:


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


# ## Graphviz
# 
# The `graphviz` utility is a collection of tools for visually graphs and directed graphs. Unfortunately, the package can be troublesome to install on laptops in a way that is compatable with many JupyterLab installations. Accordingly, the following cell is intended for use on Google Colab which provides a preinstalled version of `graphviz`.

# In[15]:


import graphviz
from graphviz import Digraph
import sys

if "google.colab" in sys.modules:

    dot = Digraph(
        node_attr = {"fontsize": "10", "shape": "square", "style": "filled"},
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
            if m.x[src, dst]() > 0:
                dot.edge(src, dst, f"rate = {rates.loc[dst, src]}\nshipped = {m.x[dst, src]()}")

    dot


# In[ ]:




