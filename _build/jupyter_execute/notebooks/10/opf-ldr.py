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
# # Optimal power flow problem with recourse actions
# 
# In this notebook we illustrate an application of the idea of linear decision rules to a two-stage optimal power flow problem in which the power of the conventional generators has to adapt automatically to balance out energy surplus/shortage due to changes in renewable resources input.
# 
# We will be working with a modified version of the [energy dispatch problem](../04/power-network.ipynb) and the [OPF problem with wind curtailment](../10/opf-wind-curtailment.ipynb). Please refer to the corresponding notebooks for the basics of power networks.

# In[1]:


# install pyomo and select solver
import sys

SOLVER = "cbc"

if "google.colab" in sys.modules:
    get_ipython().system('pip install highspy >/dev/null')
    SOLVER = "appsi_highs"


# ## Problem description
# 
# We consider a variant of the Optimal Power Flow problem in which each conventional generator $i$ commits in advance to produce a specific amount $p_i$ of energy as determined by the OPF problem assuming the renewable energy production from all solar panels and wind turbines will be equal to the forecasted one, also denoted as $\bar{p}_j$. The realized renewable energy output of generator $j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}$, however, shall deviate from their forecast values, by an amount $\Delta_j$, and result in a power production of
# 
# $$
# p_j = \bar{p}_j + \Delta_j, \quad  j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}
# $$
# 
# Then, the conventional generators need to take a _recourse action_ to make sure that the network is balanced, i.e., that the total energy production equals the total energy demand. This means that the problem has a two-stage structure:
# - first, the 'nominal' energy generation levels are set for the coal and gas units
# - then, the actual renewable energy output of the wind/solar generators is observed
# - power generation levels of the coal and gas units need to be adapted.
# 
# If we were optimizing for the average-case total cost, then a proper two-stage formulation of our problem would be as follows:
# 
# $$
# \begin{align*}
# \begin{array}{llll}
# \min \quad & \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} c_i(\bar{p}_i) +  \mathbb{E} Q(\bar{p}, \Delta) \\
# \text{s.t.} \quad & p_{i}^{\min } \leq \bar{p}_{i} \leq p_{i}^{\max } & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}
# \end{array}
# \end{align*}
# $$
# where the second stage problem is
# $$
# \begin{align*}
# \begin{array}{lllll}
# Q(\bar{p},\Delta) := &\min \quad & \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \hat{c}_i(r_i) \\
# &\text{s.t.} \quad & p_i = \bar{p}_i + r_i & \forall \, i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# && p_{i}^{\min } \leq p_{i} \leq p_{i}^{\max } & \forall \, i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# && \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = p_i - d_i & \forall \, i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# && \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = \bar{p}_i + \Delta_i - d_i & \forall \, i \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}\\
# && \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = \bar{p}_i - d_i & \forall \, i \in V \setminus (\mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}} \cup \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}) \\
# && f_{ij} =  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E \\
# && -f_{ij}^{\max} \leq f_{ij} \leq  f_{ij}^{\max}    & \forall \, (i, j) \in E\\
# && \theta_i \in \mathbb{R} & \forall \, i \in V \\
# && f_{ij} \in \mathbb{R} & \forall \, (i, j) \in E\\
# && r_i \in \mathbb{R} & \forall \, i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}},
# \end{array}
# \end{align*}
# $$
# 
# where $c_i(.)$ and $\hat{c}_i(.)$ are the cost functions related to the pre-committed $\bar{p}_i$ and changed $r_i$ amounts of energy, remembering that among $\bar{p}_i$'s, only those with $i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}$ are actual decision variables while $\bar{p}_i$, $i \in \mathcal{G}^{\text{solar}} \cup \mathcal{G}^{\text{wind}}$ are known parameters (think of them as wind and solar power forecasts).
# 
# In the [OPF problem with wind curtailment](../10/opf-wind-curtailment.ipynb) we solved a similar problem using the SAA method. More specifically, we generated a number of scenarios for the unknown parameters and combined the second-stage problems related to each of these scenarios with the single first-stage problem obtaining one large MILO problem.
# In this notebook, we are going to use a slightly different approach, namely implementing a specific type of linear decision rules. For each conventional generator, we set a recourse action, that is real-time adjustment of its power production, based on the realization of the renewable energy. More specifically, each conventional generator $i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}$ has a _participation factor_ $\alpha_i \geq 0$ which determines to which extent that generator responds to the total imbalance $\sum_j \Delta_j$. Specifically, the power production _after the recourse action_ at the conventional generator $i$ is denoted by $p_i$ and is given by
# 
# $$
# p_i := \bar{p}_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j, \qquad i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}
# $$
# 
# In the notation of the problem above, we picked the recourse action $r_i$ to be a function of the uncertain parameters $\Delta$'s rather than a decision variable, since
# 
# $$
# r_i = - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j.
# $$
# 
# The participation factor $\alpha_i \in [0,1]$ indicates the fraction of the power imbalance that generator $i$ needs to help compensate. To ensure that the power balance is satisfied, we need to have $\sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \alpha_i = 1$. Indeed, in this case, assuming the power was balanced in the first stage, i.e., $\sum_{i \in \mathcal{G}} p_i - \sum_{i \in V} d_i =0$, then the net power balance after the second stage is
# 
# $$
# \begin{align*}
# \sum_{i \in \mathcal{G}} p_i - \sum_{i \in V} d_i 
# &= \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} (\bar{p}_j + \Delta_j) + \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} (\bar{p}_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j) - \sum_{i \in V} d_i\\
# &= \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j - \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}}  \left (\sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \alpha_i \right) \Delta_j + \sum_{i \in \mathcal{G}} \bar{p}_i - \sum_{i \in V} d_i \\
# & = \sum_{i \in \mathcal{G}} \bar{p}_i - \sum_{i \in V} d_i = 0
# \end{align*}
# $$
# 
# The participation factors $\alpha_i$'s do not have to be equal for all generators and in fact, they can be optimized jointly together with the initial power levels $p_i$. Since the energy produced as recourse action is more expensive, we account for this by adding to the objective function the cost term $\sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \hat{c}_i(\alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j)$ for some cost functions $\hat{c}_i(.)$.
# 
# The resulting two-stage stochastic problem is
# $$
# \begin{align*}
# \begin{array}{llll}
# \min \quad & \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} c_i(\bar{p}_i) + \mathbb{E} Q(\bar{p}, \alpha, \Delta) \\
# \text{s.t.} \quad 
# & \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \alpha_i = 1 \\
# & p_{i}^{\min } \leq \bar{p}_{i} \leq p_{i}^{\max } & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# & \alpha_i \geq 0 & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}
# \end{array}
# \end{align*}
# $$
# where the second stage problem is
# $$
# \begin{align*}
# \begin{array}{lllll}
# Q(\bar{p},\alpha,\Delta) := &\min \quad & \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \hat{c}_i(\alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j) \\
# &\text{s.t.} \quad & p_{i}^{\min } \leq p_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j \leq p_{i}^{\max } & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# && \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = \bar{p}_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j - d_i & \forall \, i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# && \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = \bar{p}_i + \Delta_i - d_i & \forall \, i \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}\\
# && \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = \bar{p}_i - d_i & \forall \, i \in V \setminus (\mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}} \cup \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}) \\
# && f_{ij} =  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E \\
# && -f_{ij}^{\max} \leq f_{ij} \leq  f_{ij}^{\max}    & \forall (i, j) \in E\\
# && \theta_i \in \mathbb{R} & \forall i \in V \\
# && f_{ij} \in \mathbb{R}  & \forall (i, j) \in E.
# \end{array}
# \end{align*}
# $$
# 
# The only remaining question is where to take the values for $\Delta_j$'s from. One way of doing things would be to construct an uncertainty set for $\Delta_j$ and make sure that the inequality constraints hold for all realizations of $\Delta_j$ using the technique of robust counterparts/adversarial approach from Chapter 8. This would be a solid approach if we were optimizing for the worst-case value of the objective function, instead of an expectation.
# 
# However, since in this particular application we optimize for the expected value it makes sense for us to resort to the SAA method. More specifically, we sample $T$ realizations of the renewable fluctuations $\Delta$'s, denoted as $\{\Delta^s\}_{s=1,\dots,T}$, and we approximate the expectation through an empirical average across all those samples while enforcing that the constraints hold for every such realization. In this way, the resulting problem we actually solve is:
# 
# $$
# \begin{align*}
# \begin{array}{llll}
# & \min \quad & \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} c_i(\bar{p}_i) + \frac{1}{T} \sum\limits_{s = 1}^T  \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \hat{c}_i(\alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j^s)\\
# & \text{s.t.} \quad & p_{i}^{\min } \leq \bar{p}_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j^s \leq p_{i}^{\max } & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}, \, \forall s = 1, \ldots, T \\
# && \sum_{j: (i, j) \in E} f_{ij}^s - \sum_{j: (j, i) \in E} f_{ji}^s = \bar{p}_i - \alpha_i \sum_{j \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}} \Delta_j^s - d_i & \forall \, i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}, \, \forall s = 1, \ldots, T\\
# && \sum_{j: (i, j) \in E} f_{ij}^s - \sum_{j: (j, i) \in E} f_{ji}^s = \bar{p}_i + \Delta_i^s - d_i & \forall \, i \in \mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}}, \, \forall s = 1, \ldots, T\\
# && \sum_{j: (i, j) \in E} f_{ij}^s - \sum_{j: (j, i) \in E} f_{ji}^s = \bar{p}_i - d_i & \forall \, i \in V \setminus (\mathcal{G}^{\text{wind}} \cup \mathcal{G}^{\text{solar}} \cup \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}), \, \forall s = 1, \ldots, T\\
# && f_{ij}^s =  b_{ij}(\theta_i^s - \theta_j^s), & \forall \, (i, j) \in E, \, \forall s = 1, \ldots, T\\
# && -f_{ij}^{\max} \leq f_{ij}^s \leq  f_{ij}^{\max}    & \forall (i, j) \in E, \, \forall s = 1, \ldots, T\\
# && \theta_i^s \in \mathbb{R} & \forall i \in V, \, \forall s = 1, \ldots, T \\
# && f_{ij}^s \in \mathbb{R} & \forall (i, j) \in E, \, \forall s = 1, \ldots, T\\
# && \sum_{i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}} \alpha_i = 1 \\
# && p_{i}^{\min } \leq \bar{p}_{i} \leq p_{i}^{\max } & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}} \\
# && \alpha_i \geq 0 & \forall i \in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}.
# \end{array}
# \end{align*}
# $$
# 
# To recover a linear problem, we assume that the energy generation costs are modeled as:
# $$
# \begin{align*}
#     c_i(x) := c_i x, \quad \hat{c}_i(x) = 2 c_i |x|,
# \end{align*}
# $$
# where $c_i$ is the unit production cost of the $i$-th generator. This structure for the $\hat{c}_i(\cdot)$ functions means we assume that any real-time adjustment in the energy dispatch of a generator is twice as costly as a pre-scheduled unit of energy generated there.

# ## Pyomo solution

# ### Data imports
# 
# Importantly for our problem, the variable costs of producing a unit of energy per generator are stored in their corresponding 'c_var' attributes. For the rest, we are using the same data elements as in the [OPF problem with wind curtailment](../10/opf-wind-curtailment.ipynb).

# In[2]:


# Load packages
import pyomo.environ as pyo
from IPython.display import Markdown, HTML
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
import networkx as nx
import time

# Download the data
base_url = "https://raw.githubusercontent.com/mobook/MO-book/main/notebooks/10/"
nodes_df = pd.read_csv(base_url + "nodes.csv", index_col=0)
edges_df = pd.read_csv(base_url + "edges.csv", index_col=0)

# Rework the nodes and edges data into a form that is convenient for NetworkX to work with
nodes = nodes_df.set_index("node_id").T.to_dict()
edges = edges_df.set_index(edges_df["edge_id"].apply(make_tuple)).T.to_dict()
I = {"nodes": nodes, "edges": edges}

# Initialize a network for demonstration purposes
network = I


# ### SAA implementation
# 
# The following cell is the Pyomo function implementation of our final optimization problem above, with an optional argument indicating if the participation factors should be taken as uniform across all conventional generators. For ease of accounting for the second-stage costs $\hat{c}_i(\cdot)$, we assume that the function takes as an argument not only a list of scenarios for the renewable energy generation, but also the corresponding total and absolute values of the imbalances in the network.

# In[3]:


# Define an OPF problem with recourse actions for the conventional generators based on participation factors
def OPF_participationfactors(
    network,
    imbalances,
    totalimbalances,
    abstotalimbalances,
    uniformparticipationfactors=False,
):
    # Define a model
    model = pyo.ConcreteModel("OPF with participation factors")

    # Define sets
    model.T = pyo.Set(initialize=range(len(imbalances)))
    model.V = pyo.Set(initialize=network["nodes"].keys())
    model.E = pyo.Set(initialize=network["edges"].keys())
    model.SWH = pyo.Set(
        initialize=[
            i
            for i, data in network["nodes"].items()
            if data["energy_type"] in ["wind", "solar", "hydro"]
        ]
    )
    model.CG = pyo.Set(
        initialize=[
            i
            for i, data in network["nodes"].items()
            if data["energy_type"] in ["coal", "gas"]
        ]
    )
    model.NG = pyo.Set(
        initialize=[
            i
            for i, data in network["nodes"].items()
            if pd.isna(data["energy_type"])
        ]
    )

    # Declare decision variables
    model.p = pyo.Var(model.V, domain=pyo.NonNegativeReals)
    model.r = pyo.Var(model.V, model.T, domain=pyo.NonNegativeReals)
    model.alpha = pyo.Var(model.V, domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(model.V, model.T, domain=pyo.Reals)
    model.f = pyo.Var(model.E, model.T, domain=pyo.Reals)
    model.abs_total_imbalance = pyo.Param(
        model.T, domain=pyo.NonNegativeReals, initialize=abstotalimbalances
    )
    model.total_imbalance = pyo.Param(
        model.T, domain=pyo.Reals, initialize=totalimbalances
    )

    # Declare objective function including the recourse actions
    model.objective = pyo.Objective(
        expr=sum(
            data["c_var"] * model.p[i]
            for i, data in network["nodes"].items()
            if data["is_generator"]
        )
        + 1
        / len(model.T)
        * sum(
            sum(
                2
                * data["c_var"]
                * model.alpha[i]
                * model.abs_total_imbalance[t]
                for i, data in network["nodes"].items()
                if data["energy_type"] in ["coal", "gas"]
            )
            for t in model.T
        ),
        sense=pyo.minimize,
    )

    # Declare constraints
    # First-stage production levels must meet generator limits
    model.generation_upper_bound = pyo.Constraint(
        model.V, rule=lambda m, i: m.p[i] <= network["nodes"][i]["p_max"]
    )
    model.generation_lower_bound = pyo.Constraint(
        model.V, rule=lambda m, i: network["nodes"][i]["p_min"] <= m.p[i]
    )

    # Wind, solar and hydro generators have zero participation factors
    model.windsolarhydro_nopartecipationfactors = pyo.Constraint(
        model.SWH, rule=lambda m, i: m.alpha[i] == 0
    )

    # Load nodes have zero participation factors
    model.load_nopartecipationfactors = pyo.Constraint(
        model.NG, rule=lambda m, i: m.alpha[i] == 0
    )

    # Participation factors must sum to one
    model.sum_one = pyo.Constraint(
        rule=sum(model.alpha[i] for i in model.V) == 1
    )

    if uniformparticipationfactors:
        # Participation factors must be equal
        model.equal_participationfactors = pyo.Constraint(
            model.CG, rule=lambda m, i: m.alpha[i] == 1 / len(model.CG)
        )

    # Second-stage production levels must also meet generator limits
    model.power_withrecourse = pyo.Constraint(
        model.V,
        model.T,
        rule=lambda m, i, t: m.r[i, t]
        == m.p[i] - m.alpha[i] * m.total_imbalance[t],
    )
    model.generation_upper_bound_withrecourse = pyo.Constraint(
        model.CG,
        model.T,
        rule=lambda m, i, t: m.r[i, t] <= network["nodes"][i]["p_max"],
    )
    model.generation_lower_bound_withrecourse = pyo.Constraint(
        model.CG,
        model.T,
        rule=lambda m, i, t: network["nodes"][i]["p_min"] <= m.r[i, t],
    )

    # Expressions for outgoing and incoming flows
    model.outgoing_flow = pyo.Expression(
        model.V,
        model.T,
        rule=lambda m, i, t: sum(
            m.f[i, j, t] for j in model.V if (i, j) in model.E
        ),
    )
    model.incoming_flow = pyo.Expression(
        model.V,
        model.T,
        rule=lambda m, i, t: sum(
            m.f[j, i, t] for j in model.V if (j, i) in model.E
        ),
    )

    # Net power production at each node after recourse actions
    model.flow_conservation = pyo.Constraint(
        model.V,
        model.T,
        rule=lambda m, i, t: m.incoming_flow[i, t] - m.outgoing_flow[i, t]
        == m.r[i, t] + imbalances[t][i] - nodes[i]["d"],
    )

    model.susceptance = pyo.Constraint(
        model.E,
        model.T,
        rule=lambda m, i, j, t: m.f[(i, j), t]
        == network["edges"][(i, j)]["b"] * (m.theta[i, t] - m.theta[j, t]),
    )

    model.flows_upper_bound = pyo.Constraint(
        model.E,
        model.T,
        rule=lambda m, i, j, t: m.f[(i, j), t]
        <= network["edges"][(i, j)]["f_max"],
    )
    model.flows_lower_bound = pyo.Constraint(
        model.E,
        model.T,
        rule=lambda m, i, j, t: -m.f[(i, j), t]
        <= network["edges"][(i, j)]["f_max"],
    )

    # Solve the model
    solver = pyo.SolverFactory(SOLVER)
    result = solver.solve(model)

    return model


# ### Scenario generation
# Next, we generate $T=100$ scenarios in which the wind and solar production deviate from the forecasted values. Such deviations, named *imbalances*, are generated uniformly at random assuming the realized wind or solar power is between 0.5 and 1.5 times the forecasted value.
# 
# For ease of calculations, for each scenario, we define a separate data structure with the total energy imbalance and the total absolute imbalance.

# In[4]:


# Define the set of nodes with possible deviations from forecasts, i.e. those with either a wind or a solar generator
SW = {48, 53, 58, 60, 64, 65}
SW_df = nodes_df[nodes_df["node_id"].isin(SW)]

# Define the number of scenarios and the random seed
T = 100
seed = 0
rng = np.random.default_rng(seed)

# Imbalances are generated uniformly at random assuming the realized
# wind or solar power is between 0.5 and 1.5 times the forecasted value
imbalances = [
    {
        i: rng.uniform(-nodes_df["p_min"][i] / 2, nodes_df["p_min"][i] / 2)
        if i in SW
        else 0
        for i in nodes_df.index
    }
    for t in range(T)
]
totalimbalances = {
    t: sum(imbalances[t].values()) for t in range(len(imbalances))
}
abstotalimbalances = {
    t: abs(totalimbalances[t]) for t in range(len(totalimbalances))
}


# ### Perfect forecast case (no imbalances)
# 
# We first solve the optimization model in the case where the forecast for solar and wind power are perfect, meaning there is no imbalance. In this case, the recourse actions are not needed and the second stage part of the problem is trivial. 

# In[5]:


# Define trivial arrays for the case of perfect forecast and no need of recourse actions
zeroimbalances = [{i: 0 for i in nodes_df.index}]
zerototalimbalances = {0: sum(zeroimbalances[0].values())}
zeroabstotalimbalances = {0: abs(zerototalimbalances[0])}

# Solve the model
m = OPF_participationfactors(
    network,
    zeroimbalances,
    zerototalimbalances,
    zeroabstotalimbalances,
    uniformparticipationfactors=True,
)
firststagecost = sum(
    data["c_var"] * m.p[i].value
    for i, data in network["nodes"].items()
    if data["is_generator"]
)
print(f"First-stage energy production cost = {firststagecost:.2f}")
print(
    f"The optimal production levels for the conventional generators are: {[np.round(m.p[i].value, 2) for i in m.CG]}"
)


# It is key to understand how this solution would perform if there were perturbations in renewable energy production.
# 
# First of all, it is not guaranteed that it is possible to find a feasible solution in such a case. Following an approach similar to that of the [OPF problem with wind curtailment](../10/opf-wind-curtailment.ipynb), one can solve for the remaining variables when keeping the initial solution $\bar{p}$ fixed and check that with uniform participation factors, it is not possible to find a feasible flow in any of the scenarios we consider. If instead we allow for non-uniform participation factors, then this is not possible in 13 out of 100 scenarios.
# 
# Putting the feasibility issues aside for a moment, let us check how much extra cost would there be with uniform participation factors would there be on average across our scenarios. We can calculate this by taking the total imbalance and computing the cost of the recourse action to cover it, assuming that every coal and gas generator adjusts its production proportionally to the optimal participation factor previously obtained.

# In[6]:


averagerecoursecost = (
    1
    / T
    * sum(
        sum(
            2 * data["c_var"] * m.alpha[i].value * abstotalimbalances[t]
            for i, data in network["nodes"].items()
        )
        for t in range(T)
    )
)
averagetotalcost = m.objective() + 1 / T * sum(
    sum(
        2 * data["c_var"] * m.alpha[i].value * abstotalimbalances[t]
        for i, data in network["nodes"].items()
    )
    for t in range(T)
)
print(
    f"Average energy production cost due to recourse actions = {averagerecoursecost:.2f} (but including many infeasible scenarios!)"
)
print(
    f"Average total cost = {averagetotalcost:.2f} (but including many infeasible scenarios!)"
)


# ### Stochastic case (nonzero imbalances)
# 
# If we assume that the forecast for solar and wind power is not perfect, then the total energy imbalance in the network will be nonzero in each scenario. The resulting average total cost of energy production would be much higher than the deterministic scenario. This is intuitive because recourse actions are needed to cover the imbalance, and the recourse actions are much more expensive than the first-stage production-level decisions.
# 
# We now solve the two-stage stochastic optimization model that accounts for the fluctuations of solar and wind power from their forecasts, using the 100 generated scenarios. In this case, the recourse actions are still needed, but we assume fixed uniform participation factors equal to $0.1$ for all the ten conventional generators in $\in \mathcal{G}^{\text{coal}} \cup \mathcal{G}^{\text{gas}}$.

# In[7]:


m = OPF_participationfactors(
    network,
    imbalances,
    totalimbalances,
    abstotalimbalances,
    uniformparticipationfactors=True,
)
print(
    "The optimal production levels for the conventional generators are",
    [np.round(m.p[i].value, 2) for i in m.CG],
)
print(
    "The participation factors for the conventional generators are",
    [np.round(m.alpha[i].value, 2) for i in m.CG],
)

firststagecost = sum(
    data["c_var"] * m.p[i].value
    for i, data in network["nodes"].items()
    if data["is_generator"]
)
averagerecoursecost = (
    1
    / T
    * sum(
        sum(
            2 * data["c_var"] * m.alpha[i].value * m.abs_total_imbalance[t]
            for i, data in network["nodes"].items()
        )
        for t in m.T
    )
)
print(f"\nFirst-stage energy production cost = {firststagecost:.2f}")
print(
    f"Average energy production cost due to recourse actions = {averagerecoursecost:.2f}"
)
print(f"Total cost = {m.objective():.2f}")


# We see that the total average production cost is slightly higher than in the first "perfect forecast" nominal scenario, but the benefit of this newly obtained solution is that we are sure that in all scenarios we have a feasible power flow dispatch.
# 
# We argue that using this solution should be preferable: even if the 'nominal' solution had a slightly lower average production cost, we need to factor in how costly will it be for the network operator when the solution becomes infeasible. Having an infeasible network configuration means there is a risk of cascading failures and/or blackout, which besides possibly damaging the infrastructure is dramatically more expensive from a financial and societal perspective, and having a 13% chance of this happening is just unaffordable.
# 
# Next, we also optimize the participation factors $\alpha_i$'s jointly with the initial power levels, to see if we can achieve a reduction in the average total cost. We can do this using the same function as before but changing the argument ``uniformparticipationfactors`` to ``False``.

# In[8]:


m = OPF_participationfactors(
    network,
    imbalances,
    totalimbalances,
    abstotalimbalances,
    uniformparticipationfactors=False,
)
print(
    "The optimal production levels for the conventional generators are",
    [np.round(m.p[i].value, 2) for i in m.CG],
)
print(
    "The optimal participation factors for the conventional generators are",
    [np.round(m.alpha[i].value, 2) for i in m.CG],
)

firststagecost = sum(
    data["c_var"] * m.p[i].value
    for i, data in network["nodes"].items()
    if data["is_generator"]
)
averagerecoursecost = (
    1
    / T
    * sum(
        sum(
            2 * data["c_var"] * m.alpha[i].value * m.abs_total_imbalance[t]
            for i, data in network["nodes"].items()
        )
        for t in m.T
    )
)
print(f"\nFirst-stage energy production cost = {firststagecost:.2f}")
print(
    f"Average energy production cost due to recourse actions = {averagerecoursecost:.2f}"
)
print(f"Total cost = {m.objective():.2f}")


# This energy dispatch is about 1.4% cheaper than the solution with uniform participation factors. It might seem like a small difference but in view of the high volumes of energy produced and consumed, say at the national level, this makes a huge difference.
