#!/usr/bin/env python
# coding: utf-8

# # Energy dispatch problem
# 
# To meet the energy demand, power plants run day and night across the country to produce electricy from a variety of sources such as fossil fuels and renewable energy. On the short-time scale, the best operating levels for electric power plants are derived every 15 minutes by solving the so-called *Optimal Power Flow (OPF)* model. The OPF model is an optimization problem with the objective of minimizing the total energy dispatching cost, while ensuring that the generation meets the total energy demand. Furthermore, the model takes into account many constraints, among which operational and physical constraints. 
# 

# In[1]:


get_ipython().system('curl -s https://raw.githubusercontent.com/jckantor/MO-book/main/tools/_mobook.py -o mobook.py')
import mobook
mobook.setup_pyomo()
mobook.setup_glpk()
mobook.svg()


# ## Background: Power networks and power flow physics
# 
# We model the nation-wide transmission power network as a directed graph $G=(V, E)$, where $V$ represents the set of nodes (e.g., cities, industrial districts, power plants) and $E$ denotes the set of directed edges (e.g., physical transmission lines). 
# 
# Each node $i \in V$ has a power injection $p_i$ and demand $d_i$. The set of nodes are separated into *generator* and *load* nodes. The set of generators $\mathcal{G} \subseteq V$ corresponds to the nodes $i \in V$ for which $p_i \geq 0$ and $d_i = 0$. Each generator $i \in \mathcal{G}$ has a minimum $p_i^{\min}$ and maximum $p_i^{\max}$ production capacity. The set of load nodes $\mathcal{D} \subseteq V$ corresponds to the nodes for which $p_i = 0$ and $d_i \geq 0$. The load nodes thus correspond to the places where electricity is being consumed, e.g., cities and industrial districts. We say that supply and demand is *matched* if $\sum_{i \in V} p_i - d_i = 0$. Since we cannot store electricity in large volumes, supply must meet demand at all times, hence adjusting to it every 15 minutes by solving the OPF.
# 
# Each edge $(i, j) \in E$ carries a *power flow* $f_{ij} \in R$ and has a capacity $f_{ij}^\max \geq 0$, i.e., the maximum power flow that it may carry. Note that our choice to model a *directed* graph is to make the modeling of the network easier. In particular, a directed edge $(i,j)$ may carry a 'negative' flow $f_{ij} < 0$, which implies that there is flow going from $j$ to $i$ where $f_{ji} = -f_{ij}$. The capacity does not depend on the direction of the flow, implying that the flow capacity constraints are given by $|f_{ij}| = |f_{ji}| \leq f_{ij}^\max$.
# 
# One crucial difference of power flow problems compared to typical network flow problems is that the power flows cannot be controlled directly. Instead, as you might recall from high-school physics, the power flows are determined by the laws of electricity, which we will now present as the *power flow equations*. Ignore for a moment the flow capacity constraints. Let $\theta_{i} \in \mathbb{R}$ denote the *phase angle* of node $i$. For each edge $(i,j)$, let $b_{ij} > 0$ denote the *line susceptance*. Assuming that supply and demand is matched, i.e., $\sum_{i=1}^{n} p_i - d_i = 0$, the power flows $\mathbf{f} \in \mathbb{R}^{m}$ and phase angles $\mathbf{\theta} \in \mathbb{R}^{n}$ are obtained by solving the following linear system of equations:
# 
# $$\begin{align}
# p_i - d_i &= \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji}, & \forall \, i \in V,\\
#   f_{ij} &=  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E.
# \end{align}$$
# 
# The first set of constraints ensures flow conservation and the second set of constrations captures the flow dependency on susceptances and angle differences. The DC power flow equations admit a unique power flow solution $\mathbf{f}$ given matched power injections $\mathbf{p}$ and demand $\mathbf{d}$. 
# 
# For a given matched supply and demand vector $\mathbf{p}$ and $\mathbf{d}$, we can compute the power flows on the network by solving the linear equations as described above. There are exactly $|V|$ and $|E|$ equations for the $\theta_i$ variables and $f_{ij}$ variables, meaning that this system of equations admit a solution.

# ## Optimal Power Flow
# We assumed above that the power injections $\mathbf{p}$ were given. However, in practice, the power injections need to be determined for each generator in the power network, where some types of generators may be cheaper than others. Moreover, we need to take into account operational constraints, such as the maximum flow and generator limits. 
# 
# On the short-time scale, the power injections are calculated for each generator by solving the so-called *Optimal Power Flow (OPF)* problem. The goal of the OPF problem is to determine a solution $(\mathbf{p}, \mathbf{f}, \mathbf{\theta})$ with minimal costs such that:
# - Supply meets demand
# - Line capacity constraints are met
# - Generator capacity constraints are met
# 
# Let $c_i > 0$ be the cost associated with the production of a unit energy by generator $i$. Then, the OPF problem can be formulated as
# 
# $$\begin{align}
# \begin{array}{llll}
# \max        & \sum_{i \in V} c_i p_i \\
# \mbox{s.t.} & \sum_{j: (i, j) \in E} f_{ij} - \sum_{j: (j, i) \in E} f_{ji} = p_i - d_i & \forall \, i \in V,\\
# & f_{ij} =  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E, \\
#             & |f_{ij}| \leq  f_{ij}^{\max}    & \forall (i, j) \in E,\\
#             & p_{i}^{\min } \leq p_{i} \leq p_{i}^{\max }         & \forall i \in V, \\
#             & p_i \in \mathbb{R}_{\geq 0}        & \forall i \in V,                \\
#             & \theta_i \in \mathbb{R} & \forall i \in V, \\
#             & f_{ij} \in \mathbb{R}                 & \forall (i, j) \in E                \\
# \end{array}
# \end{align}$$
# 
# For simplicity, you may assume that all load nodes do not produce energy, i.e., $p_i = p_i^{\min} = p_i^\max = 0$ for all $i \in \mathcal{D}$. You may therefore model $p_i$ as decision variables for all nodes (both generator and load nodes). Similarly, you may assume that all generator nodes have no demand, i.e., $d_i = 0$ for all $i \in \mathcal{G}$.
# 
# To summarize, the decision variables in the OPF problem are: 
# - $p_i$ power injections
# - $\theta_i$ phase angles
# - $f_{ij}$ power flows
# 
# All the other quantities are instance dependent parameters.

# ## Data
# 

# You will solve the OPF problem on the IEEE-118 power network, which is a standard test network consisting of 118 nodes and 179 edges. In the following, we will load the data into the notebook and provide a description of the data. Make sure to run the setup code below. We will first describe the data which are initially provided as `pandas.DataFrames`. We will then provide a new network data structure to easily access the data for the implementation of the optimization model.

# ### Setup code

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

# Load GLPK solver
solver = pyo.SolverFactory('glpk')


# ### Data import

# In[3]:


# Download the data
nodes_df = pd.read_csv('https://gist.githubusercontent.com/leonlan/0fce00d4ea330fd11d49f35c71ba5d8c/raw/146d180aad2ef67d451cd8745ccca98a6d2db68a/final_nodes.csv', index_col=0)
edges_df = pd.read_csv('https://gist.githubusercontent.com/leonlan/0fce00d4ea330fd11d49f35c71ba5d8c/raw/5ce01cafcdf659e7f073fda31df731a1550683ef/final_edges.csv', index_col=0)

# Read all instances
nodes = nodes_df.groupby("instance").apply(lambda data: data.set_index("node_id").T.to_dict())
edges = edges_df.set_index(edges_df["edge_id"].apply(make_tuple)).T.to_dict()
I = [{"nodes": n, "edges": edges} for n in nodes]

# Initialize a network for demonstration purposes
network = I[0]
example_nodes = nodes_df[nodes_df.instance == 0]
example_edges = nodes_df[nodes_df.instance == 0]


# ### Network data

# In[4]:


def visualize_network(network, edge_flows=None, ax=None):
    """Visualize a network instance, highlighting the generators in orange and the load buses in green."""
    plt.figure(figsize=[12, 10])
    g = nx.DiGraph(network["edges"].keys())
    pos = nx.layout.kamada_kawai_layout(g, weight=None)

    color_mapping = {'solar': '#ffcb36',
     'wind': 'white',
     'hydro': '#a5efff',
     'coal': '#686868',
     'gas': '#00ab4e',
     np.nan: '#b6b6b6'}
              
    vertex2color = {i: color_mapping[data['energy_type']] for i, data in network['nodes'].items()}
    v2c_list = [vertex2color[i] for i in g.nodes] # Order based on networkx

    nodes = nx.draw_networkx_nodes(g, pos, node_size=250, node_color=v2c_list, linewidths=2,)
    edges = nx.draw_networkx_edges(g, pos, width=2, edge_color='#595959',)

    # Gives node colors
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#595959")
    ax.set_axis_off()


# In[5]:


visualize_network(network)


# The IEEE-118 network has 18 generators, of which:
# - 2 hydro plants (cyan)
# - 4 coal plants (dark gray)
# - 4 solarfarms (yellow)
# - 2 windmills (white)
# - 6 gas plants (green)
# 
# Moreover, the network has 100 load nodes (gray). Each node has the following parameters: 
# - `node_id`
# - `d`: demand
# - `p_min`: minimum production capacity
# - `p_max`: maximum production capacity
# - `c_var`: variable production costs
# - `is_generator`: boolean value to indicate whether the node is a generator or not
# - `energy_type`: the type of energy used for electricity production

# All generator nodes can filtered by the `is_generator` parameter. All generators have a zero demand $d_i=0$. For the renewable energy sources (i.e., hydro, solar and wind) there are no variable costs `c_var`. For solar and wind, the production is fixed, i.e., `p_min = p_max`, meaning that all available solar and wind energy must be produced.

# In[6]:


example_nodes[example_nodes.is_generator]


# For the load nodes, the only important parameter is the demand $d_i \geq 0$. All other parameters are either zero, `False`, or `NaN`.

# In[7]:


example_nodes[~example_nodes.is_generator]


# ### Edges

# The IEEE-118 network has 179 edges. Each edge has the following parameters:
# - `edge_id`: the edge indices
# - `b`: the line susceptance, and
# - `f_max`: the maximum edge capacity.

# In[8]:


edges_df


# ### Network data structure
# When implementing optimization problems, it is often not the easiest way to access data directly through `pandas.DataFrames`. We have therefore provided a new type of data structure that will help to more easily implement an optimization problem. This data structure is also found in Exercise 5 & 6 from Computer Lab 1, which is as follows:
# ```
# network = {"nodes": {i: {"attr1":  1, ...},
#                        ...
#                       },
#              "edges": {(i, j): {"attr1":  4, "attr2": 1, ...},
#                       ...
#                       }
#             }
# ```
# In other words, each network consists of a dictionary, that contains a dictionary for nodes and edges, which again holds a dictionary for each node and edge. It might a be little bit confusing at first, but this data structure allows to easily access the parameters that you need for a specific constraint. For instance,

# In[9]:


network['nodes'][1]['d']


# gives the demand of node `1`. To filter on generator nodes, you can use the following: 

# In[10]:


[i for i, data in network['nodes'].items() if data['is_generator']]


# ### Instances
# Since the OPF problem is solved every 15 minutes in practice, you will be given `24 * 4 = 96` network instances that need to be solved, hence solving an entire day's worth of OPF problems. The first instance thus corresponds to the power generation within time window `[00:00, 00:15]`, the second instance corresponds to `[00:15, 00:30]`, and so on. The data takes into account a realistic demand and (renewable energy) generation pattern. We assume that the decisions in each time window are independent of the previous and subsequent time windows, so every 15 minutes a new OPF instance is solved independently.
# 
# The network instances are stored in the variable `I` as a list and can be accessed using indexing, i.e., `I[0]` gives the first network instance and `I[95]` gives the 96th network instance.

# ---

# # Solving OPF
# Observe that the stated OPF problem contains absolute decision values. We first rewrite the OPF problem into a linear optimization problem.
# 
# $$\begin{align}
# \begin{array}{llll}
# \min        & \sum_{i \in V} c_i p_i \\
# \mbox{s.t.} & \sum_{j: (i, j) \in E} f^+_{ij} - f^-_{ij} - \sum_{j: (j, i) \in E} f^+_{ji} - f_{ji}^- = p_i - d_i & \forall \, i \in V,\\
# & f^+_{ij} - f^-_{ij} =  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E, \\
#             & f^+_{ij} + f^-_{ij} \leq  f_{ij}^{\max}    & \forall (i, j) \in E,\\
#             & p_{i}^{\min } \leq p_{i} \leq p_{i}^{\max }         & \forall i \in V, \\
#             & p_i \in \mathbb{R}_{\geq 0}        & \forall i \in V,                \\
#             & \theta_i \in \mathbb{R} & \forall i \in V, \\
#             & f_{ij}^+, f_{ij}^- \in \mathbb{R}                 & \forall (i, j) \in E                \\
# \end{array}
# \end{align}$$
# 
# We then implement the model using `pyomo` and solve it for all instances `I[0]` to `I[95]`, reporting the average objective value across all the 96 instances.

# In[11]:


def OPF1(network):
    """
    Input:
    - network: a dictionary containing:
      - a dictionary of nodes with a dictionary of attributes
      - a dictionary of edges with a dictionary of attributes

    Output:
    - power_generation: a dictionary containing the power generation for each node
    - power_flows: a dictionary containing the power flow for each edge
    """
    
    # Define a model
    model = pyo.ConcreteModel("Q1")

    # Declare decision variables
    model.p = pyo.Var(network["nodes"], domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(network["nodes"], domain=pyo.Reals)
    model.fabs = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.fp = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.fm = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)

    # Declare objective value
    @model.Objective(sense=pyo.maximize)
    def objective(model):
        return sum(data["c_var"] * model.p[i] for i, data in network["nodes"].items() if data["is_generator"])
    
    @model.Expression(network["nodes"])
    def outgoing_flow(m, i):
        return sum(m.fp[i, j] - m.fm[i, j] for j in network["nodes"] if (i, j) in network["edges"])
    
    @model.Expression(network["nodes"])
    def incoming_flow(m, i):
        return sum(m.fp[j, i] - m.fm[j, i] for j in network["nodes"] if (j, i) in network["edges"])
    
    @model.Constraint(network["nodes"])
    def flow_conservation(m, i):
        return m.outgoing_flow[i] - m.incoming_flow[i] == m.p[i] - network["nodes"][i]["d"]
    
    @model.Constraint(network["edges"])
    def susceptance(m, i, j):
        return m.fp[i, j] - m.fm[i, j] == network["edges"][(i, j)]["b"] * (m.theta[i] - m.theta[j])
    
    @model.Constraint(network["edges"])
    def abs_flow(m, *e):
        return m.fabs[e] == m.fp[e] + m.fm[e]
    
    @model.Constraint(network["edges"])
    def flows_upper_bound(m, *e):
        return m.fabs[e] <= network["edges"][e]["f_max"]
    
    @model.Constraint(network["nodes"])
    def generation_upper_bound(m, i):
        return m.p[i] <= network["nodes"][i]["p_max"]
    
    @model.Constraint(network["nodes"])
    def generation_lower_bound(m, i):
        return network["nodes"][i]["p_min"] <= m.p[i]
    
    # Solve
    result = solver.solve(model)

    return model.objective()


# In[12]:


print(f"The average objective value over all instances is: {np.mean([OPF1(instance) for instance in I])}")


# # Strict fossil fuel policy pt.1
# Gas and coal plants emit CO2, while renewable energy sources are carbon neutral. For this reason, the Dutch government has decided to constrain the number of active fossil-fuel-based power plants for the generation of electricity. More specifically, a maximum of 2 gas plants and 1 coal plant may be *active* during a single OPF instance. Any plant that is set *inactive* for a specific instance cannot produce any electricity. 
# 
# We first write down the new model. To this end, we introduce new decision variables $x_i, i\in V$ to the model, which indicate whether a generator $i$ is active or inactive.
# 
# $$\begin{align}
# \begin{array}{llll}
# \min        & \sum_{i \in V} c_i p_i \\
# \mbox{s.t.} & \sum_{j: (i, j) \in E} f^+_{ij} - f^-_{ij} - \sum_{j: (j, i) \in E} f^+_{ji} - f_{ji}^- = p_i - d_i & \forall \, i \in V,\\
# & f^+_{ij} - f^-_{ij} =  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E, \\
# & f^{abs}_{ij} = f^+_{ij} + f^-_{ij}, & \forall \, (i, j) \in E, \\
#             & f_{ij}^{abs} \leq  f_{ij}^{\max}    & \forall (i, j) \in E,\\
#             & p_{i}^{\min } x_i \leq p_{i} \leq p_{i}^{\max } x_i        & \forall i \in V, \\
#             & \sum_{i \in \mathcal{G}_{\text{gas}}} x_i \leq 2 & \\
#             & \sum_{i \in \mathcal{G}_{\text{coal}}} x_i \leq 1 &  \\
#             & p_i \in \mathbb{R}_{\geq 0}        & \forall i \in V,                \\
#             & \theta_i \in \mathbb{R} & \forall i \in V, \\
#             & f_{ij} \in \mathbb{R}                 & \forall (i, j) \in E                \\
#             & x_i  \in \{0, 1\} & \forall i \in V\\
# \end{array}
# \end{align}$$
# 
# We then implement the new model using `pyomo` and solve it for all instances `I[0]` to `I[95]`, reporting the average objective value across the instances.

# In[13]:


def OPF2(network):
    """
    Input:
    - network: a dictionary containing:
      - a dictionary of nodes with a dictionary of attributes
      - a dictionary of edges with a dictionary of attributes

    Output:
    - power_generation: a dictionary containing the power generation for each node
    - power_flows: a dictionary containing the power flow for each edge
    """
    
    # Define a model
    model = pyo.ConcreteModel("Q2")

    # Declare decision variables
    model.p = pyo.Var(network["nodes"], domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(network["nodes"], domain=pyo.Reals)
    model.fabs = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.fp = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.fm = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.x = pyo.Var(network["nodes"], domain=pyo.Binary)

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(model):
        return sum(data["c_var"] * model.p[i] for i, data in network["nodes"].items() if data["is_generator"])

    # Declare constraints
    @model.Expression(network["nodes"])
    def outgoing_flow(m, i):
        return sum(m.fp[i, j] - m.fm[i, j] for j in network["nodes"] if (i, j) in network["edges"])
    
    @model.Expression(network["nodes"])
    def incoming_flow(m, i):
        return sum(m.fp[j, i] - m.fm[j, i] for j in network["nodes"] if (j, i) in network["edges"])
    
    @model.Constraint(network["nodes"])
    def flow_conservation(m, i):
        return m.outgoing_flow[i] - m.incoming_flow[i] == m.p[i] - network["nodes"][i]["d"]
    
    @model.Constraint(network["edges"])
    def susceptance(m, i, j):
        return m.fp[i, j] - m.fm[i, j] == network["edges"][(i, j)]["b"] * (m.theta[i] - m.theta[j])
    
    @model.Constraint(network["edges"])
    def abs_flows(m, *e):
        return m.fabs[e] == m.fp[e] + m.fm[e]
    
    @model.Constraint(network["edges"])
    def flows_upper_bound(m, *e):
        return m.fabs[e] <= network["edges"][e]["f_max"]

    @model.Constraint(network["nodes"])
    def generation_upperound(m, i):
        return m.p[i] <= network["nodes"][i]["p_max"] * m.x[i]
    
    @model.Constraint(network["nodes"])
    def generation_lower_bound(m, i):
        return network["nodes"][i]["p_min"] * m.x[i] <= m.p[i]
    
    @model.Constraint()
    def max_gas_plants(model):
        return sum(model.x[i] for i, data in network["nodes"].items() if data['energy_type'] == 'gas') <= 2
    
    @model.Constraint()
    def max_coal_plants(model):
        return sum(model.x[i] for i, data in network["nodes"].items() if data['energy_type'] == 'coal') <= 1

    # Solve
    result = solver.solve(model)

    return model.objective()


# In[14]:


print(f"The average objective value over all instances is: {np.mean([OPF2(instance) for instance in I])}")


# # Strict fossil fuel policy pt.2
# The restriction on the number of gas and coal plants may pose a threat to the availability of electricity production when renewable energy sources fail to deliver. For this reason, the grid operators have decided to slightly change the constraint that was introduced for OPF2. If the total production of energy from renewable energy sources (i.e., solar, wind and hydro) is above 1000, then the number of gas and coal plants is restricted to 2 and 1, respectively. Otherwise, the restriction on the number of gas and coal plants is lifted. These constraints can be modeled as *either-or* constraints. 
#   
# We first write down the new model, using big-$M$ constraints to model the either-or constraint.
# 
# $$\begin{align}
# \begin{array}{llll}
# \min        & \sum_{i \in V} c_i p_i \\
# \mbox{s.t.} & \sum_{j: (i, j) \in E} f^+_{ij} - f^-_{ij} - \sum_{j: (j, i) \in E} f^+_{ji} - f_{ji}^- = p_i - d_i & \forall \, i \in V,\\
# & f^+_{ij} - f^-_{ij} =  b_{ij}(\theta_i - \theta_j), & \forall \, (i, j) \in E, \\
# & f^{abs}_{ij} = f^+_{ij} + f^-_{ij}, & \forall \, (i, j) \in E, \\
#             & f_{ij}^{abs} \leq  f_{ij}^{\max}    & \forall (i, j) \in E,\\
#             & p_{i}^{\min } x_i \leq p_{i} \leq p_{i}^{\max } x_i        & \forall i \in V, \\
#             & \sum_{i \in \mathcal{G}_{\text{gas}}} x_i \leq 2 + (1-y)M_0 & \\
#             & \sum_{i \in \mathcal{G}_{\text{coal}}} x_i \leq 1 + (1-y)M_1 &  \\
#             & \sum_{i \in \mathcal{G}_\text{solar}} p_i + \sum_{i \in \mathcal{G}_\text{wind}} p_i + \sum_{i \in \mathcal{G}_\text{hydro}} p_i \leq 1000 + yM_2&\\
#             & p_i \in \mathbb{R}_{\geq 0}        & \forall i \in V,                \\
#             & \theta_i \in \mathbb{R} & \forall i \in V, \\
#             & f_{ij} \in \mathbb{R}                 & \forall (i, j) \in E                \\
#             & x_i  \in \{0, 1\} & \forall i \in V\\
#             & y \in \{0, 1\} &\\
# \end{array}
# \end{align}$$
# 
# We now implement the new model using `pyomo` and solve it for all instances `I[0]` to `I[95]`, reporting the average objective value across the instances.

# In[15]:


def OPF3(network):
    """
    Input:
    - network: a dictionary containing:
      - a dictionary of nodes with a dictionary of attributes
      - a dictionary of edges with a dictionary of attributes

    Output:
    - power_generation: a dictionary containing the power generation for each node
    - power_flows: a dictionary containing the power flow for each edge
    """
    
    # Define a model
    model = pyo.ConcreteModel("Q3")

    # Declare decision variables
    model.p = pyo.Var(network["nodes"], domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(network["nodes"], domain=pyo.Reals)
    model.fabs = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.fp = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.fm = pyo.Var(network["edges"], domain=pyo.NonNegativeReals)
    model.x = pyo.Var(network["nodes"], domain=pyo.Binary)
    model.y = pyo.Var(domain=pyo.Binary)

    # Big-Ms
    max_total_renewable_production = sum(data['p_max'] for i, data in network["nodes"].items() if data['energy_type'] in ['solar', 'wind', 'hydro'])
    M = [4, 3, max_total_renewable_production - 1000]

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(data["c_var"] * m.p[i] for i, data in network["nodes"].items() if data["is_generator"])

    # Declare constraints
    @model.Expression(network["nodes"])
    def outgoing_flow(m, i):
        return sum(m.fp[i, j] - m.fm[i, j] for j in network["nodes"] if (i, j) in network["edges"])
    
    @model.Expression(network["nodes"])
    def incoming_flow(m, i):
        return sum(m.fp[j, i] - m.fm[j, i] for j in network["nodes"] if (j, i) in network["edges"])
    
    @model.Constraint(network["nodes"])
    def flow_conservation(m, i):
        return m.outgoing_flow[i] - m.incoming_flow[i] == m.p[i] - network["nodes"][i]["d"]
    
    @model.Constraint(network["edges"])
    def susceptance(m, i, j):
        return m.fp[i, j] - m.fm[i, j] == network["edges"][(i, j)]["b"] * (m.theta[i] - m.theta[j])

    @model.Constraint(network["edges"])
    def abs_flows(m, *e):
        return m.fabs[e] == m.fp[e] + m.fm[e]
    
    @model.Constraint(network["edges"])
    def flows_upper_bound(m, *e):
        return m.fabs[e] <= network["edges"][e]["f_max"]

    @model.Constraint(network["nodes"])
    def generation_upper_bound(m, i):
        return m.p[i] <= network["nodes"][i]["p_max"] * m.x[i]
    
    @model.Constraint(network["nodes"])
    def generation_lower_bound(m, i):
        return network["nodes"][i]["p_min"] * m.x[i] <= m.p[i]
    
    @model.Constraint()
    def max_gas_plants(m):
        return sum(m.x[i] for i, data in network["nodes"].items() if data['energy_type'] == 'gas') <= 2 + (1 - m.y) * M[0]
    
    @model.Constraint()
    def max_coal_plants(m):
        return sum(m.x[i] for i, data in network["nodes"].items() if data['energy_type'] == 'coal') <= 1 + (1 - m.y) * M[1]

    @model.Constraint()
    def renewable_energy_production(m):
        return sum(m.p[i] for i, data in network["nodes"].items() if data['energy_type'] in ['solar', 'wind', 'hydro']) <= 1000 + m.y * M[2]
    
    # Solve
    result = solver.solve(model)

    return  model.objective()


# In[16]:


print(f"The average objective value over all instances is: {np.mean([OPF3(instance) for instance in I])}")


# # Comparing the three models
# For all three implemented models, plot the objective values for all instances. Explain the differences between the objective values in view of the different feasible regions.

# In[17]:


objs = [[OPF1(instance) for instance in I],
        [OPF2(instance) for instance in I],
        [OPF3(instance) for instance in I]]


# In[18]:


plt.plot(objs[0], color='blue', label='OPF1')
plt.plot(objs[1], color='green', label='OPF2')
plt.plot(objs[2], color='orange', label='OPF3')
plt.title('Optimal objective values over all instances and models')
plt.xlabel('Instance number')
plt.ylabel('Optimal objective value')
plt.legend()
plt.show()


# The goal of the OPF problem is to minimize the total costs of dispatching energy. In what follows, we refer to the earlier described models as OPF1, OPF2 and OPF3, respectively.
# 
# OPF1 is the original OPF formulation, whereas OPF2 and OPF3 are restricted versions of OPF1. This means that the feasible region of OPF1 is at least as large as OPF2 and OPF3, where we may assume that $x_i = 1$ for all the generators $i$.
# 
# If we let $F_1, F_2, F_3$ denote the feasible region of OPF1, OPF2 and OPF3, then we observe that
# which explains that the optimal objective value of OPF1 always remains below the optimal objectives values of OPF2 and OPF3.
# 
# The most important observation to make is that based on the problem descriptions, we would expect OPF2 to take on the objective values of OPF1 or OPF3, depending on which of the either-or-constraints are activated. 
# - OPF3 uses expensive generators, and possibly not all the renewable energy
# - OPF2 does only uses 1000 renewable energy power, because then it may keep using all the gas and coal generators. 
# - OPF1 uses all renewable energy at all times, because it has the flexibility to use all generators in order to mitigate operational restrictions due to line flows.

# In[ ]:




