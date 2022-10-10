#!/usr/bin/env python
# coding: utf-8

# # Transportation and Allocation
# 
# This notebook presents a transportation model to optimally allocate the delivery of a commodity from multiple sources to multiple destinations. The model invites a discussion of the pitfalls in optimizing a global objective for customers who may have an uneven share of the resulting benefits, then through model refinement arrives at a group cost-sharing plan to delivery costs.
# 
# Didactically, notebook presents techniques for Pyomo modeling and reporting including: 
# 
# * `pyo.Expression` decorator
# * Accessing the duals (i.e., shadow prices)
# * Methods for reporting the solution and duals.
#     * Pyomo `.display()` method for Pyomo objects
#     * Manually formatted reports
#     * Pandas 
#     * Graphviz for display of results as a directed graph.
#    

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/jckantor/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Problem: Distributing Gasoline to Franchise Operators
# 
# YaYa Gas-n-Grub is franchisor and operator for a network of regional convenience stores selling gasoline and convenience items in the United States. Each store is individually owned by a YaYa Gas-n-Grub franchisee who pays feea to the franchisor for services.
# 
# Gasoline is delivered by truck from regional distribution terminals. Each delivery truck carries 8,000 gallons delivered at a fixed charge of 700 dollars per delivery, or 8.75 cents per gallon. The franchise owners are eager to reduce delivery costs to boost profits.
# 
# YaYa Gas-n-Grub decides to accept proposals from other distribution terminals, "A" and "B", to supply the franchise operators. Rather than a fixed fee per delivery, they proposed pricing based on location. But they already have existing customers, "A" and "B" can only provide a limited amount of gasoline to new customers totaling 100,000 and 80,000 gallons respectively. The only difference between the new suppliers and the current supplier is the delivery charge.
# 
# The following chart shows the pricing of gasoline delivery in cents/gallon.
# 
# | Franchisee <br> &nbsp; | Demand <br> &nbsp; |  Terminal A <br> 100,000| Terminal B <br> 80,000 | Current Supplier <br> 500,000 |
# | :-------- | ------------: | :---------: | :-------: | :--------: |
# | Alice | 30,000 | 8.3 | 10.2 | 8.75 |
# | Badri  | 40,000 | 8.1 | 12.0 | 8.75 |
# | Cara  | 50,000 | 8.3 | - | 8.75 |
# | Dan   | 80,000 | 9.3 | 8.0 |  8.75 |
# | Emma  | 30,000 | 10.1 | 10.0 | 8.75 |
# | Fujita | 45,000 | 9.8 | 10.0 | 8.75 |
# | Grace | 80,000 | -  | 8.0 | 8.75 |
# | Helen | 18,000 | 7.5 | 10.0 | 8.75 |
# | **TOTAL**| **313,000**| |  | | |
# 
# The franchisor and operator of YaYa Gas-n-Grub has the challenge of allocating the gasoline delivery to minimize the cost to the franchise owners. The following model will present a global objective to minimize the total cost of delivery to all franchise owners. 

# ## Model 1: Minimize Total Delivery Cost
# 
# The decision variables for this example are labeled $x_{d, s}$ where subscript $d \in 1, \dots, n_d$ refers to the destination of the delivery and subscript $s \in 1, \dots, n_s$ to the source. The value of $x_{d,s}$ is the volume of gasoline shipped to destination $d$ from source $s$.
# 
# Given the cost rate $r_{d, s}$ for shipping one unit of goods from $d$ to $s$, the objective is to minimize the total cost of transporting gasoline from the sources to the destinations as given by
# 
# $$
# \begin{align*}
# \text{objective: total delivery cost}\qquad\min & \sum_{d=1}^{n_d} \sum_{s=1}^{n_s} r_{d, s} x_{d, s} \\
# \end{align*}
# $$
# 
# subject to meeting the demand requirements, $D_d$, at all destinations, and satisfying the supply constraints, $S_s$, at all sources.
# 
# $$
# \begin{align*}
# \text{demand constraints}\qquad\sum_{s=1}^{n_s} x_{d, s} & = D_d & \forall d\in 1, \dots, n_d \\
# \text{supply constraints}\qquad\sum_{d=1}^{n_d} x_{d, s} & \leq S_s & \forall s\in 1, \dots, n_s \\
# \end{align*}
# $$
# 

# ## Data Entry
# 
# The data is stored into Pandas DataFrame and Series objects. Note the use of a large rates to avoid assigning shipments to destination, source pairs not allowed by the problem statement.

# In[2]:


import pandas as pd
from IPython.display import HTML, display

rates = pd.DataFrame(
    [
        ["Alice", 8.3, 10.2, 8.75],
        ["Badri", 8.1, 12.0, 8.75],
        ["Cara", 8.3, 100.0, 8.75],
        ["Dan", 9.3, 8.0, 8.75],
        ["Emma", 10.1, 10.0, 8.75],
        ["Fujita", 9.8, 10.0, 8.75],
        ["Grace", 100, 8.0, 8.75],
        ["Helen", 7.5, 10.0, 8.75],
    ],
    columns=["Destination", "Terminal A", "Terminal B", "Current Supplier"],
).set_index("Destination")

demand = pd.Series(
    {
        "Alice": 30000,
        "Badri": 40000,
        "Cara": 50000,
        "Dan": 20000,
        "Emma": 30000,
        "Fujita": 45000,
        "Grace": 80000,
        "Helen": 18000,
    },
    name="demand",
)

supply = pd.Series(
    {"Terminal A": 100000, "Terminal B": 80000, "Current Supplier": 500000},
    name="supply",
)

display(HTML("<br><b>Gasoline Supply (Gallons)</b>"))
display(supply.to_frame())

display(HTML("<br><b>Gasoline Demand (Gallons)</b>"))
display(demand.to_frame())

display(HTML("<br><b>Transportation Rates (US cents per Gallon)</b>"))
display(rates)


# ## Pyomo Model 1: Minimize Total Delivery Cost
# 
# The pyomo model is an implementation of the mathematical model described above. The sets and indices have been designated with more descriptive symbols readability and maintenance. 

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
    def total_cost(m):
        return sum(
            m.Rates[dst, src] * m.x[dst, src] for dst, src in m.DESTINATIONS * m.SOURCES
        )

    @m.Expression(m.DESTINATIONS)
    def cost_to_destination(m, dst):
        return sum(m.Rates[dst, src] * m.x[dst, src] for src in m.SOURCES)

    @m.Expression(m.DESTINATIONS)
    def shipped_to_destination(m, dst):
        return sum(m.x[dst, src] for src in m.SOURCES)

    @m.Expression(m.SOURCES)
    def shipped_from_source(m, src):
        return sum(m.x[dst, src] for dst in m.DESTINATIONS)

    @m.Constraint(m.SOURCES)
    def supply_constraint(m, src):
        return m.shipped_from_source[src] <= supply[src]

    @m.Constraint(m.DESTINATIONS)
    def demand_constraint(m, dst):
        return m.shipped_to_destination[dst] == demand[dst]

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("cbc").solve(m)

    return m


m = transport(supply, demand, rates / 100)

results = pd.DataFrame(
    {dst: {src: m.x[dst, src]() for src in m.SOURCES} for dst in m.DESTINATIONS}
).T
results["current costs"] = 700 * demand / 8000
results["contract costs"] = pd.Series(
    {dst: m.cost_to_destination[dst]() for dst in m.DESTINATIONS}
)
results["savings"] = results["current costs"].round(1) - results[
    "contract costs"
].round(1)
results["contract rate"] = round(results["contract costs"] / demand, 4)
results["marginal cost"] = pd.Series(
    {dst: m.dual[m.demand_constraint[dst]] for dst in m.DESTINATIONS}
)

print(f"Old Delivery Costs = $ {sum(demand)*700/8000}")
print(f"New Delivery Costs = $ {m.total_cost()}")
display(results)

results.plot(y="savings", kind="bar")
model1_results = results


# ## Model 2: Minimize Cost Rate for Franchise Owners
# 
# Minimizing total costs provides no guarantee that individual franchise owners will benefit equally, or in fact benefit at all, from minimizing total costs. In this example neither "Emma" or "Fujita" would save any money on delivery costs, and the majority of savings goes to just two of the franchisees.  Without a better distribution of the benefits there may be little enthusiasm among the franchisees to adopt change.
# 
# This observation motivates an attempt at a second model. In this case the objective is minimize a common rate for the cost of gasoline distribution.
# 
# $$
# \begin{align*}
# \text{objective: common distribution rate}\qquad\min \rho \\
# \end{align*}
# $$
# 
# subject to meeting the demand and supply constraints, $S_s$, at all sources.
# 
# $$
# \begin{align*}
# \text{common cost rate}\qquad\sum_{s=1}^{n_s} r_{d, s} x_{d, s} & \leq \rho D_d & \forall d\in 1, \dots, n_d\\
# \text{demand constraints}\qquad\sum_{s=1}^{n_s} x_{d, s} & = D_d & \forall s\in 1, \dots, n_d \\
# \text{supply constraints}\qquad\sum_{d=1}^{n_d} x_{d, s} & \leq S_s & \forall s\in 1, \dots, n_s \\
# \end{align*}
# $$
# 
# The following Pyomo model implements this formulation.

# In[4]:


import pyomo.environ as pyo


def transport(supply, demand, rates):
    m = pyo.ConcreteModel()

    m.SOURCES = pyo.Set(initialize=rates.columns)
    m.DESTINATIONS = pyo.Set(initialize=rates.index)

    m.x = pyo.Var(m.DESTINATIONS, m.SOURCES, domain=pyo.NonNegativeReals)
    m.rate = pyo.Var()

    @m.Param(m.DESTINATIONS, m.SOURCES)
    def Rates(m, dst, src):
        return rates.loc[dst, src]

    @m.Objective(sense=pyo.minimize)
    def delivery_rate(m):
        return m.rate

    @m.Expression(m.DESTINATIONS)
    def cost_to_destination(m, dst):
        return sum(m.Rates[dst, src] * m.x[dst, src] for src in m.SOURCES)

    @m.Expression()
    def total_cost(m):
        return sum(m.cost_to_destination[dst] for dst in m.DESTINATIONS)

    @m.Constraint(m.DESTINATIONS)
    def rate_to_destination(m, dst):
        return m.cost_to_destination[dst] == m.rate * demand[dst]

    @m.Expression(m.DESTINATIONS)
    def shipped_to_destination(m, dst):
        return sum(m.x[dst, src] for src in m.SOURCES)

    @m.Expression(m.SOURCES)
    def shipped_from_source(m, src):
        return sum(m.x[dst, src] for dst in m.DESTINATIONS)

    @m.Constraint(m.SOURCES)
    def supply_constraint(m, src):
        return m.shipped_from_source[src] <= supply[src]

    @m.Constraint(m.DESTINATIONS)
    def demand_constraint(m, dst):
        return m.shipped_to_destination[dst] == demand[dst]

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("cbc").solve(m)

    return m


m = transport(supply, demand, rates / 100)

results = round(
    pd.DataFrame(
        {dst: {src: m.x[dst, src]() for src in m.SOURCES} for dst in m.DESTINATIONS}
    ).T,
    1,
)
results["current costs"] = 700 * demand / 8000
results["contract costs"] = round(
    pd.Series({dst: m.cost_to_destination[dst]() for dst in m.DESTINATIONS}), 1
)
results["savings"] = results["current costs"].round(1) - results[
    "contract costs"
].round(1)
results["contract rate"] = round(results["contract costs"] / demand, 4)
results["marginal cost"] = pd.Series(
    {dst: m.dual[m.demand_constraint[dst]] for dst in m.DESTINATIONS}
)

print(f"Old Delivery Costs = $ {sum(demand)*700/8000}")
print(f"New Delivery Costs = $ {m.total_cost()}")
display(results)

results.plot(y="savings", kind="bar")


# ## Model 3: Minimize Total Cost for a Cost-Sharing Plan
# 
# The prior two models demonstrated some practical difficulties in realizing the benefits of a cost optimization plan. Model 1 will likely fail in a franchisor/franchisee arrangement because the realized savings would be fore the benefit of a few. 
# 
# Model 2 was an attempt to remedy the problem by solving for an allocation of deliveries that would lower the cost rate that would be paid by each franchisee directly to the gasoline distributors. Perhaps surprisingly, the resulting solution offered no savings to any franchisee. Inspecting the data shows the source of the problem is that two franschisees, "Emma" and "Fujita", simply have no lower cost alternative than the current supplier. Therefore finding a distribution plan with direct payments to the distributors that lowers everyone's cost is an impossible task.
# 
# The third model addresses this problem with a plan to share the cost savings among the franchisees. In this plan, the franchisor would collect delivery fees from the franchisees to pay the gasoline distributors. The optimization objective returns to the problem to minimizing total delivery costs, but then adds a constraint that defines a common cost rate to charge all franchisees. By offering a benefit to all parties, the franchisor offers incentive for group participation in contracting for gasoline distribution services.
# 
# 
# $$
# \begin{align*}
# \text{objective: total delivery cost}\qquad\min  C \\
# \end{align*}
# $$
# 
# subject to meeting the demand and supply constraints, $S_s$, at all sources.
# 
# $$
# \begin{align*}
# \text{total cost}\qquad \sum_{d=1}^{n_d} \sum_{s=1}^{n_s} r_{d, s} x_{d, s} & = C \\
# \text{uniform cost sharing rate}\qquad\sum_{s=1}^{n_s} r_{d, s} x_{d, s} & = \rho D_d & \forall d\in 1, \dots, n_d\\
# \text{demand constraints}\qquad\sum_{s=1}^{n_s} x_{d, s} & = D_d & \forall d\in 1, \dots, n_d \\
# \text{supply constraints}\qquad\sum_{d=1}^{n_d} x_{d, x} & \leq S_s & \forall s\in 1, \dots, n_s \\
# \end{align*}
# $$

# In[5]:


import pyomo.environ as pyo


def transport(supply, demand, rates):
    m = pyo.ConcreteModel()

    m.SOURCES = pyo.Set(initialize=rates.columns)
    m.DESTINATIONS = pyo.Set(initialize=rates.index)

    m.x = pyo.Var(m.DESTINATIONS, m.SOURCES, domain=pyo.NonNegativeReals)
    m.rate = pyo.Var()

    @m.Param(m.DESTINATIONS, m.SOURCES)
    def Rates(m, dst, src):
        return rates.loc[dst, src]

    @m.Objective()
    def total_cost(m):
        return sum(
            m.Rates[dst, src] * m.x[dst, src] for dst, src in m.DESTINATIONS * m.SOURCES
        )

    @m.Expression(m.DESTINATIONS)
    def cost_to_destination(m, dst):
        return m.rate * demand[dst]

    @m.Constraint()
    def allocate_costs(m):
        return sum(m.cost_to_destination[dst] for dst in m.DESTINATIONS) == m.total_cost

    @m.Expression(m.DESTINATIONS)
    def shipped_to_destination(m, dst):
        return sum(m.x[dst, src] for src in m.SOURCES)

    @m.Expression(m.SOURCES)
    def shipped_from_source(m, src):
        return sum(m.x[dst, src] for dst in m.DESTINATIONS)

    @m.Constraint(m.SOURCES)
    def supply_constraint(m, src):
        return m.shipped_from_source[src] <= supply[src]

    @m.Constraint(m.DESTINATIONS)
    def demand_constraint(m, dst):
        return m.shipped_to_destination[dst] == demand[dst]

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("cbc").solve(m)

    return m


m = transport(supply, demand, rates / 100)

results = round(
    pd.DataFrame(
        {dst: {src: m.x[dst, src]() for src in m.SOURCES} for dst in m.DESTINATIONS}
    ).T,
    1,
)
results["current costs"] = 700 * demand / 8000
results["contract costs"] = round(
    pd.Series({dst: m.cost_to_destination[dst]() for dst in m.DESTINATIONS}), 1
)
results["savings"] = results["current costs"].round(1) - results[
    "contract costs"
].round(1)
results["contract rate"] = round(results["contract costs"] / demand, 4)
results["marginal cost"] = pd.Series(
    {dst: m.dual[m.demand_constraint[dst]] for dst in m.DESTINATIONS}
)

print(f"Old Delivery Costs = $ {sum(demand)*700/8000}")
print(f"New Delivery Costs = $ {m.total_cost()}")
display(results)

results.plot(y="savings", kind="bar")
model3_results = results


# ## Comparing Model Results
# 
# The following charts demonstrate the difference in outcomes for Model 1 and Model 3 (Model 2 was left out as entirely inadequate). The group cost-sharing arrangement produces the same group savings, but distributes the benefits in a manner likely to be more acceptable to the majority of participants.

# In[6]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
alpha = 0.45

model1_results.plot(y=["savings"], kind="bar", ax=ax[0], color="g", alpha=alpha)
model1_results.plot(y="savings", marker="o", ax=ax[0], color="g", alpha=alpha)

model3_results.plot(y="savings", kind="bar", ax=ax[0], color="r", alpha=alpha)
model3_results.plot(y="savings", marker="o", ax=ax[0], color="r", alpha=alpha)
ax[0].legend(["Model 1", "Model 3"])
ax[0].set_title("delivery costs by franchise")

model1_results.plot(y=["contract rate"], kind="bar", ax=ax[1], color="g", alpha=alpha)
model1_results.plot(y="contract rate", marker="o", ax=ax[1], color="g", alpha=alpha)

model3_results.plot(y="contract rate", kind="bar", ax=ax[1], color="r", alpha=alpha)
model3_results.plot(y="contract rate", marker="o", ax=ax[1], color="r", alpha=alpha)
ax[1].set_ylim(0.07, 0.09)
ax[1].legend(["Model 1", "Model 3"])
ax[1].set_title("delivery cost rate by franchise")


# ## Didactics: Reporting Solutions
# 
# Pyomo models can produce considerable amounts of data that must be summarized and presented for analysis and decision making. In this application, for example, the individual franchise owners receive differing amounts of savings which is certain to result in considerable discussion and possibly negotiation with the franchisor. 
# 
# The following cells demonstrate techniques for extracting and displaying information generated by a Pyomo model. 

# ### Pyomo `.display()` method
# 
# Pyomo provides a default `.display()` method for most Pyomo objects. The default display is often sufficient for model reporting requirements, particularly when initially developing a new application.

# In[7]:


# display elements of sets
m.SOURCES.display()
m.DESTINATIONS.display()


# In[8]:


# display elements of an indexed parameter
m.Rates.display()


# In[9]:


# display elements of Pyomo Expression
m.shipped_to_destination.display()


# In[10]:


m.shipped_from_source.display()


# In[11]:


# display Pyomo Objective
m.total_cost.display()


# In[12]:


# display indexed Pyomo Constraint
m.supply_constraint.display()
m.demand_constraint.display()


# In[13]:


# display Pyomo decision variables
m.x.display()


# ### Manually formatted reports
# 
# Following solution, the value associated with Pyomo objects are returned by calling the object as a function. The following cell demonstrates the construction of a custom report using Python f-strings and Pyomo methods.

# In[14]:


# Objective report
print("\nObjective: cost")
print(f"cost = {m.total_cost()}")

# Constraint reports
print("\nConstraint: supply_constraint")
for src in m.SOURCES:
    print(
        f"{src:12s}  {m.supply_constraint[src]():8.2f}  {m.dual[m.supply_constraint[src]]:8.2f}"
    )

print("\nConstraint: demand_constraint")
for dst in m.DESTINATIONS:
    print(
        f"{dst:12s}  {m.demand_constraint[dst]():8.2f}  {m.dual[m.demand_constraint[dst]]:8.2f}"
    )

# Decision variable reports
print("\nDecision variables: x")
for src in m.SOURCES:
    for dst in m.DESTINATIONS:
        print(f"{src:12s} -> {dst:12s}  {m.x[dst, src]():8.2f}")
    print()


# ### Pandas
# 
# The Python Pandas library provides a highly flexible framework for data science applications. The next cell demonstrates the translation of Pyomo object values to Pandas DataFrames

# In[15]:


suppliers = pd.DataFrame(
    {
        src: {
            "supply": supply[src],
            "shipped": m.supply_constraint[src](),
            "sensitivity": m.dual[m.supply_constraint[src]],
        }
        for src in m.SOURCES
    }
).T

display(suppliers)

customers = pd.DataFrame(
    {
        dst: {
            "demand": demand[dst],
            "shipped": m.demand_constraint[dst](),
            "sensitivity": m.dual[m.demand_constraint[dst]],
        }
        for dst in m.DESTINATIONS
    }
).T

display(customers)

shipments = pd.DataFrame(
    {dst: {src: m.x[dst, src]() for src in m.SOURCES} for dst in m.DESTINATIONS}
).T
display(shipments)
shipments.plot(kind="bar")


# ### Graphviz
# 
# The `graphviz` utility is a collection of tools for visually graphs and directed graphs. Unfortunately, the package can be troublesome to install on laptops in a way that is compatible with many JupyterLab installations. Accordingly, the following cell is intended for use on Google Colab which provides a preinstalled version of `graphviz`.

# In[16]:


import sys

import graphviz
from graphviz import Digraph

if "google.colab" in sys.modules:

    dot = Digraph(
        node_attr={"fontsize": "10", "shape": "rectangle", "style": "filled"},
        edge_attr={"fontsize": "10"},
    )

    for src in m.SOURCES:
        label = (
            f"{src}"
            + f"\nsupply = {supply[src]}"
            + f"\nshipped = {m.supply_constraint[src]()}"
            + f"\nsens  = {m.dual[m.supply_constraint[src]]}"
        )
        dot.node(src, label=label, fillcolor="lightblue")

    for dst in m.DESTINATIONS:
        label = (
            f"{dst}"
            + f"\ndemand = {demand[dst]}"
            + f"\nshipped = {m.demand_constraint[dst]()}"
            + f"\nsens  = {m.dual[m.demand_constraint[dst]]}"
        )
        dot.node(dst, label=label, fillcolor="gold")

    for src in m.SOURCES:
        for dst in m.DESTINATIONS:
            if m.x[dst, src]() > 0:
                dot.edge(
                    src,
                    dst,
                    f"rate = {rates.loc[dst, src]}\nshipped = {m.x[dst, src]()}",
                )

    display(dot)


# In[ ]:




