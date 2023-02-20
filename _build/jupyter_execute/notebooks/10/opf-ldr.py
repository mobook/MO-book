#!/usr/bin/env python
# coding: utf-8

# ```{index} single: application; energy systems
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} pandas dataframe
# ```
# ```{index} network optimization
# ```
# ```{index} stochastic optimization
# ```
# ```{index} SAA
# ```
# ```{index} linear decision rules
# ```
# 
# # Two-stage energy dispatch optimization using linear decision rules
# 
# This notebook illustrates a two-stage stochastic optimization problem in the context of energy systems where there are recourse actions are modeled as linear decision rules. Like in the [previous notebook](../10/opf-wind-curtailment.ipynb), the goal of the optimization problem is to ensure that power demand meets supply while taking into account both wind and solar fluctuations and physical infrastructure constraints. 
# 
# For more explanation about the network topology and background information about the OPF problem **read first** the [energy dispatch problem](../04/power-network.ipynb) from the Chapter 4 and the [OPF problem with wind curtailment](../10/opf-wind-curtailment.ipynb) from the Chapter 10.  

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Package and data import

# In[2]:


# Load packages
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
import networkx as nx
import time

# Load solver
solver = pyo.SolverFactory("cbc")
gsolver = pyo.SolverFactory("gurobi")

# Download the data
nodes_df = pd.read_csv("nodes.csv", index_col=0)
edges_df = pd.read_csv("edges.csv", index_col=0)

# Read data
nodes = nodes_df.set_index("node_id").T.to_dict()
edges = edges_df.set_index(edges_df["edge_id"].apply(make_tuple)).T.to_dict()
network = {"nodes": nodes, "edges": edges}


# ### Network data
# 
# This notebook uses the same network as in [energy dispatch problem](../04/power-network.ipynb) and hence the same data structure. In particular, the nodes and edges of the network are stored in the `nodes` and `edges` dataframes, respectively. The `edges` dataframe contains the following columns:   
# * `edge id`: the unique identifier of the edge, describing the node pair `(i,j)` of the edge;
# * `f_max`: describing the maximum power flow capacity of the edge; and
# * `b`: the susceptance of the edge.

# In[4]:


edges_df


# The network includes 18 generators of differen type, which is described in the `energy_type` field. We distinguish between conventional generators (coal, gas) and renewable generators (hydro, solar, wind). Every conventional generator node has two parameters:
# 
# - `c_fixed`, describing the activation cost of the conventional generators;
# - `c_var`, describing the unit cost of producing energy for each conventional generator
# 
# Renewable generators are assumed to have zero marginal cost and zero activation cost. Nodes `64` and `65` correspond to the two wind generators that this notebook focuses on.

# In[28]:


nodes_df


# In[3]:


nodes_df[nodes_df.is_generator]


# ## Problem description
# ### Optimal power flow problem with recourse actions via participation factors
# 
# We now consider a variant of the OPF problem in which each conventional generator $i$ commits in advance to produce a specific amount $p_i$ of energy as determined by the OPF problem assuming the renewable energy production from all solar panels and wind turbines will be equal to the forecasted one, also denoted as $p_j$. Assume the realized renewable energy output of generator $j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}$ deviates from their forecasted values, say by an amount $\Delta_j$, and results in a power production of
# 
# $$
# r_j = p_j + \Delta_j.
# $$
# 
# Then, the conventional generators need take a _recourse action_ to make sure that the network is balanced, i.e., that the total energy production equals the total energy demand. The recourse action at a conventional generator consists of a real-time adjustment of its power production and is modeled as follows. Each conventional generator $i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}$ has a _participation factor_ $\alpha_i \geq 0$ which determines in which that generator responds to the total imbalance $\sum_j \Delta_j$. Specfically, the power production _after the recourse action_ at generator $i$ is denoted by $r_i$ and is given by
# 
# $$
# r_i = p_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j.
# $$
# 
# The participation factor $\alpha_i \in [0,1]$ indicates the fraction of the power imbalance that generator $i$ needs to help compensate. To ensure that the power balance is satisfied, we need to have $\sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \alpha_i = 1$. Indeed, in this case, assuming the power was balanced in the first stage, i.e., $\sum_{i \in \mathcal{G}} p_i - \sum_{i \in V} d_i =0$, then the net power balance after the second stage is
# 
# $$
# \begin{align*}
# \sum_{i \in \mathcal{G}} r_i - \sum_{i \in V} d_i 
# &= \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} (p_j + \Delta_j) + \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} (p_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j) - \sum_{i \in V} d_i\\
# &= \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j - \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}}  \left (\sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \alpha_i \right) \Delta_j + \sum_{i \in \mathcal{G}} p_i - \sum_{i \in V} d_i \\
# & = \sum_{i \in \mathcal{G}} p_i - \sum_{i \in V} d_i = 0
# \end{align*}
# $$
# 
# Nonetheless, the participation factors do not have to be equal for all generators and in fact, these factors can be optimized jointly together with the initial power levels $p_i$. Since the energy produced as recourse action is twice as expensive, we account for this by adding to the objective function the cost term $\sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} 2 \cdot c^{\text{var}}_i \cdot \alpha_i \sum_{j \in V} \Delta_j$.

# In[ ]:


# Define an OPF problem with recourse actions for the conventional generators based on participation factors

def UC_participationfactors(network, imbalances, totalimbalances, abstotalimbalances, uniformparticipationfactors=False):

  # Define a model
  model = pyo.ConcreteModel("OPF with participation factors")

  # Define sets
  model.T = pyo.Set(initialize=range(len(imbalances)))
  model.V = pyo.Set(initialize=network["nodes"].keys())
  model.E = pyo.Set(initialize=network["edges"].keys())
  model.SWH = pyo.Set(initialize=[i for i, data in network["nodes"].items() if (data['energy_type'] == 'wind' or data['energy_type'] == 'solar' or data['energy_type'] == 'hydro')])
  model.CG = pyo.Set(initialize=[i for i, data in network["nodes"].items() if (data['energy_type'] == 'coal' or data['energy_type'] == 'gas')])
  model.NG = pyo.Set(initialize=[i for i, data in network["nodes"].items() if pd.isna(data['energy_type'])])
  
  # Declare decision variables
  # model.x = pyo.Var(model.V, domain=pyo.Binary)
  model.p = pyo.Var(model.V, domain=pyo.NonNegativeReals)
  model.r = pyo.Var(model.V, model.T, domain=pyo.NonNegativeReals)
  model.alpha = pyo.Var(model.V, domain=pyo.NonNegativeReals)
  model.theta = pyo.Var(model.V, model.T, domain=pyo.Reals)
  model.f = pyo.Var(model.E, model.T, domain=pyo.Reals)
  model.abs_total_imbalance = pyo.Param(model.T, domain=pyo.NonNegativeReals, initialize=abstotalimbalances)
  model.total_imbalance = pyo.Param(model.T, domain=pyo.Reals, initialize=totalimbalances)

  # Declare objective function including the recourse actions

  model.objective = pyo.Objective(expr = sum(data["c_var"] * model.p[i] for i, data in network["nodes"].items() if data["is_generator"]) +
                                  2/len(model.T) * sum(sum(data["c_var"] * model.alpha[i] * model.abs_total_imbalance[t] for i, data in network["nodes"].items() if data['energy_type'] in ['coal', 'gas']) for t in model.T),
                                    sense=pyo.minimize)

  # Declare constraints
  
  # First-stage production levels must meet generator limits
  model.generation_upper_bound = pyo.Constraint(model.V, rule=lambda m, i: m.p[i] <= network["nodes"][i]["p_max"])
  model.generation_lower_bound = pyo.Constraint(model.V, rule=lambda m, i: network["nodes"][i]["p_min"] <= m.p[i])

  # Wind, solar and hydro generators have zero participation factors
  model.windsolarhydro_nopartecipationfactors = pyo.Constraint(model.SWH, rule=lambda m, i: m.alpha[i] == 0)

  # Load nodes have zero participation factors
  model.load_nopartecipationfactors = pyo.Constraint(model.NG, rule=lambda m, i: m.alpha[i] == 0)

  # Participation factors must sum to one
  model.sum_one = pyo.Constraint(rule = sum(model.alpha[i] for i in model.V) == 1)

  if uniformparticipationfactors:
    # Participation factors must be equal
    model.equal_participationfactors = pyo.Constraint(model.CG, rule = lambda m, i: m.alpha[i] == 1/len(model.CG))

  # Second-stage production levels must also meet generator limits
  model.power_withrecourse = pyo.Constraint(model.V, model.T, rule=lambda m, i, t: m.r[i, t] == m.p[i] - m.alpha[i] * m.total_imbalance[t])
  model.generation_upper_bound_withrecourse = pyo.Constraint(model.CG, model.T, rule=lambda m, i, t: m.r[i, t] <= network["nodes"][i]["p_max"])
  model.generation_lower_bound_withrecourse = pyo.Constraint(model.CG, model.T, rule=lambda m, i, t: network["nodes"][i]["p_min"] <= m.r[i, t])

  # Expressions for outgoing and incoming flows
  model.outgoing_flow = pyo.Expression(model.V, model.T, rule=lambda m, i, t: sum(m.f[i, j, t] for j in model.V if (i, j) in model.E))
  model.incoming_flow = pyo.Expression(model.V, model.T, rule=lambda m, i, t: sum(m.f[j, i, t] for j in model.V if (j, i) in model.E))

  # Net power production at each node after recourse actions
  model.flow_conservation = pyo.Constraint(model.V, model.T, rule=lambda m, i, t: m.incoming_flow[i, t] - m.outgoing_flow[i, t] == m.r[i, t] + imbalances[t][i] - nodes[i]["d"])
  model.susceptance = pyo.Constraint(model.E, model.T, rule=lambda m, i, j, t: m.f[(i, j), t] == network["edges"][(i, j)]["b"] * (m.theta[i, t] - m.theta[j, t]))

  model.flows_upper_bound = pyo.Constraint(model.E, model.T, rule=lambda m, i, j, t: m.f[(i, j), t] <= network["edges"][(i, j)]["f_max"])
  model.flows_lower_bound = pyo.Constraint(model.E, model.T, rule=lambda m, i, j, t: - m.f[(i, j), t] <= network["edges"][(i, j)]["f_max"])

  # Solve the model
  solver = pyo.SolverFactory("cbc")
  result = solver.solve(model)
  if result.solver.status != 'ok':
    print(f"Solver status: {result.solver.status}, {result.solver.termination_condition}")
  
  return model


# We generate $T=100$ scenarios in which the wind and solar production deviate from the forecasted values. Such deviations, named *imbalances*, are generated uniformly at random assuming the realized wind or solar power is between 0.5 and 1.5 times the forecasted value.

# In[ ]:


# Define the set of nodes with possible deviations from forecasts, i.e. those with either a wind or a solar generator
SW = {48, 53, 58, 60, 64, 65}
SW_df = nodes_df[nodes_df['node_id'].isin(SW)]

# Define the number of scenarios and the random seed
T = 100
seed = 0
rng = np.random.default_rng(seed)

# Imbalances are generated uniformly at random assuming the realized wind or solar power is between 0.5 and 1.5 times the forecasted value
imbalances = [{i: rng.uniform(-nodes_df['p_min'][i]/2, nodes_df['p_min'][i]/2) if i in SW else 0 for i in nodes_df.index} for t in range(T)]
totalimbalances = {t: sum(imbalances[t].values()) for t in range(len(imbalances))}
abstotalimbalances = {t: abs(totalimbalances[t]) for t in range(len(totalimbalances))}


# We first solve the optimization model in the case where the forecast for solar and wind power are perfect, meaning there is no imbalance. In this case the recourse actions are not needed and the second stage part of the problem is trivial. 
# 
# If we now test the performance of this static solution over the sampled scenarios, we can calculate the average energy production cost due to recourse actions assuming uniform participation factors, but this average is *misleading* because **it does not account for the fact that for many of the scenarios there is no power flow dispatch**.

# In[ ]:


zeroimbalances = [{i: 0 for i in nodes_df.index}]
zerototalimbalances = {0: sum(zeroimbalances[0].values())}
zeroabstotalimbalances = {0: abs(totalimbalances[0])}

m = UC_participationfactors(network, zeroimbalances, zerototalimbalances, zeroabstotalimbalances, True)
print(f'First-stage energy production cost = {sum(data["c_var"] * m.p[i].value for i, data in network["nodes"].items() if data["is_generator"]):.2f}')
print('The optimal production levels for the conventional generators are',[np.round(m.p[i].value,2) for i in m.CG])
print('The optimal participation factors for the conventional generators are',[np.round(m.alpha[i].value,2) for i in m.CG])
print(f'Average energy production cost due to recourse actions = {2/T * sum(sum(data["c_var"] * m.alpha[i].value * abstotalimbalances[t] for i, data in network["nodes"].items()) for t in range(T)):.2f} (but including many infeasible scenarios!)')


# We now solve the two-stage optimization model in the case where the realization of solar and wind power deviate from their forecasts. In this case, the recourse actions are needed, but we assume fixed uniform participation factors equal to $0.1$ for all the ten conventional generators in $\in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}$. We solve this model using SAA after having sampled $T=100$ realizations of the wind and solar power production. 

# In[ ]:


m = UC_participationfactors(network, imbalances, totalimbalances, abstotalimbalances, True)
print('The optimal production levels for the conventional generators are',[np.round(m.p[i].value,2) for i in m.CG])
print('The optimal participation factors for the conventional generators are',[np.round(m.alpha[i].value,2) for i in m.CG])
print(f'First-stage energy production cost = {sum(data["c_var"] * m.p[i].value for i, data in network["nodes"].items() if data["is_generator"]):.2f}')
print(f'Average energy production cost due to recourse actions = {2/T * sum(sum(data["c_var"] * m.alpha[i].value * m.abs_total_imbalance[t] for i, data in network["nodes"].items()) for t in m.T):.2f}')
print(f'Total cost = {m.objective():.2f}')


# If we instead let the model optimize the participation factors jointly with the initial power levels, we can achieve a reduction in the average total cost.

# In[ ]:


m = UC_participationfactors(network, imbalances, totalimbalances, abstotalimbalances)
print('The optimal production levels for the conventional generators are',[np.round(m.p[i].value,2) for i in m.CG])
print('The optimal participation factors for the conventional generators are',[np.round(m.alpha[i].value,2) for i in m.CG])
print(f'First-stage energy production cost = {sum(data["c_var"] * m.p[i].value for i, data in network["nodes"].items() if data["is_generator"]):.2f}')
print(f'Average energy production cost due to recourse actions = {2/T * sum(sum(data["c_var"] * m.alpha[i].value * m.abs_total_imbalance[t] for i, data in network["nodes"].items()) for t in m.T):.2f}')
print(f'Total cost = {m.objective():.2f}')

