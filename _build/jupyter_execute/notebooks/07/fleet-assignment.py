#!/usr/bin/env python
# coding: utf-8

# # Fleet assignment problem

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Problem description
# 
# Given a set of flights to be flown, an airline company needs to determine the specific route flown by each airplane in the most cost-effective way. Clearly, the airline company should try to use as fewer airplanes as possible, but the same airplane can operate two subsequent flights only if the time interval between the arrival of the first flight and the departure of the next flight is longer than or equal to one hour.
# 
# The task of the airline operations team is to determine the minimum number of airplanes needed to operate the given list of flights. This problem is known as the **fleet assignment problem** or **aircraft rotation problem**.

# In[324]:


import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from numpy.random import RandomState

# python generator to create departure and arrival times from single airport
def generate_flights(N_flights=30, min_duration=1, max_duration=4, max_departure=24, seed=0):
    rs = RandomState(seed)
    for flight in range(N_flights):
        end_flight = max_departure + 1
        while end_flight > max_departure:
            start_flight = np.floor(max_departure * rs.rand())
            end_flight = start_flight + 3 * np.ceil((max_duration + 1 - min_duration) * rs.rand()) 
        yield flight + 1, int(start_flight), int(end_flight)

# generate flight data
FlightData = pd.DataFrame([[flight, departure, arrival] for flight, departure, arrival in generate_flights()])
FlightData.columns = ["Flight", "Departure", "Arrival"]
FlightData.set_index("Flight", inplace=True)


# ## Visualize Flight Data

# In[335]:


# visualize flight data
def draw_flights(FlightData):
    bar_style = {'alpha':1.0, 'lw':10, 'solid_capstyle':'butt'}
    text_style = {'fontsize': 7, 'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, xlabel="Departure/Arrival Time", title="FlightData")
    ax.get_yaxis().set_visible(False)
    for flight, row in FlightData.iterrows():
        departure, arrival = row
        ax.plot([departure, arrival], [flight]*2, 'gray', **bar_style)
        ax.text((departure + arrival)/2, flight, flight, **text_style)
    return ax

draw_flights(FlightData)


# ## Find All Feasible Flight Pairs

# In[337]:


# create an iterator to produce pairs of flight numbers
flight_pairs = list(filter(
    lambda pair: FlightData.loc[pair[0], "Arrival"] < FlightData.loc[pair[1], "Departure"],
    [[i, j] for i in FlightData.index for j in FlightData.index if i != j]
))


# ## Visualize Feasible Flight Pairs as a Directed Graph

# In[338]:


ax = draw_flights(FlightData)
for flight1, flight2 in flight_pairs:
    arr = FlightData.loc[flight1, "Arrival"]
    dep = FlightData.loc[flight2, "Departure"]
    c = 'r' if dep - arr <= 1 else 'k'
    ax.plot([arr, dep], [flight1, flight2], color=c, lw=1, alpha=0.4)


# ## Directed Graph
# 
# Networkx includes useful tools for the modeling and analysis of directed graphs. For this application, we use Networkx to compute the input edges and output edges for each flight node. These computations are straightforward and could easily be done with a few lines of Python, but Networkx makes the overall model more expressive and easy to read.

# In[330]:


# create DiGraph

DG = nx.DiGraph()

# display all nodes
for flight in FlightData.index:
    DG.add_node(flight)

# display all edges
for flight1, flight2 in flight_pairs:
    DG.add_edge(flight1, flight2)


# ## Pyomo Model

# In[339]:


m = pyo.ConcreteModel("Fleet Assignment")
  
m.FLIGHTS = pyo.Set(initialize=FlightData.index)
m.PAIRS = pyo.Set(initialize=flight_pairs)

m.x = pyo.Var(m.PAIRS, domain=pyo.Binary)
m.source = pyo.Var(m.FLIGHTS, domain=pyo.Binary)
m.sink = pyo.Var(m.FLIGHTS, domain=pyo.Binary)

@m.Expression(m.FLIGHTS)
def sum_in_edges(m, flight):
    return m.source[flight] + sum(m.x[in_flight, flight] for in_flight, _ in DG.in_edges(flight))

@m.Expression(m.FLIGHTS)
def sum_out_edges(m, flight):
    return m.sink[flight] + sum(m.x[flight, out_flight] for _, out_flight in DG.out_edges(flight))

@m.Constraint(m.FLIGHTS)
def one_airplane_for_each_flight(m, flight):
    return m.sum_in_edges[flight] == 1

@m.Constraint(m.FLIGHTS)
def balance(m, flight): 
    return m.sum_in_edges[flight] == m.sum_out_edges[flight]

@m.Objective(sense=pyo.minimize)
def minimize_airplanes(m):
    return pyo.summation(m.source)

pyo.SolverFactory("cbc").solve(m)

print(f"Minimum airplanes required = {m.minimize_airplanes()}")

ax = draw_flights(FlightData)
for flight1, flight2 in flight_pairs:
    if m.x[flight1, flight2]() == 1:
        arr = FlightData.loc[flight1, "Arrival"]
        dep = FlightData.loc[flight2, "Departure"]
        c = 'r' if dep - arr <= 1 else 'k'
        ax.plot([arr, dep], [flight1, flight2], color=c, lw=4, alpha=0.4)


# ## Allocate Aircraft

# In[435]:


aircraft = 0
at_risk = 0
for flight in FlightData.index:
    if m.source[flight]() == 1:
        aircraft += 1
        print(f"\nAircraft {aircraft:2d}")
        print(f"    Flight {flight:2d}: depart {FlightData.loc[flight, 'Departure']:2d}", end="")
        print(f" --> arrive {FlightData.loc[flight, 'Arrival']:2d}")
        f = flight
        while m.sink[f]() != 1:
            for _, g in DG.out_edges(f):
                if m.x[f, g]() == 1:
                    break
            dt = FlightData.loc[g, "Departure"] - FlightData.loc[f, "Arrival"]
            if dt <= 1:
                print(f"  **", end="")
                at_risk += 1
            else:
                print(f"    ", end="")
            print(f"Flight {g:2d}: depart {FlightData.loc[g, 'Departure']:2d}", end="")
            print(f" --> arrive {FlightData.loc[g, 'Arrival']:2d}")
            f = g
            
print(f"\n**{at_risk} flights at risk of delay")


# ## Reformulation the fleet assignment as a network problem
# 
# The fleet assignment problem can be formulated and solved as a MILP. The idea of the MILP formulation is that for each aircraft to construct a feasible path in a graph where the flights are nodes with indices $1, \ldots, F$, and there is one source node $0$ and a sink node $F + 1$ such that only arcs can be used between nodes that can be operated after each other. Denote the set of flight indices by $\mathcal{F} = \left\{ 1, \ldots, F \right\}$, and the set of aircraft indices as $\mathcal{M} = \{ 1, \ldots, M\} $.
# 
# The set of arcs $\mathcal{A} \subseteq \mathcal{F} \cup \{0\} \times \mathcal{F} \cup \{F+1\}$ that can be used by each aircraft is then:
# 
# $$
# \mathcal{A} = \{ a = (f_1, f_2): \text{arrival of } f_1 \text{at least 1h departure of } f_2, \ f_1, f_2 \in \mathcal{F} \} \cup \{ a = (0, f), \ f \in \mathcal{F} \} \cup \{ a = (f, F+1), \ f \in \mathcal{F} \},
# $$
# 
# where $0$ and $F + 1$ play the role of dummy flights/nodes.
# 
# In this context, our decision variables shall be $x_{a, i} \in \{0, 1\}$, $a \in \mathcal{A}$, $i \in \mathcal{M}$ indicating whether the pair of flights $(f_1, f_2) = a$ is operated by aircraft $i$. Furthermore, we use the binary variables $y_i \in \{0, 1\}$, for $i \in \mathcal{M}$, to check whether a given aircraft $i$ is used or not. The corresponding MILP is then:
# 
# \begin{align}
#     \min\limits_{x_{a, i}, y_i} \quad & \sum\limits_{i \in \mathcal{M}} y_i \label{eq:71a}\tag{a}\\
#     \text{s.t.} \quad 
#     & \sum\limits_{g \in \mathcal{F} \cup \{ 0, F+1\} ~:~ a = (g, f) \in \mathcal{A}} x_{a, i} = \sum\limits_{h \in \mathcal{F} \cup \{ 0, F+1\} ~:~ a' = (f, h) \in \mathcal{A}} x_{a', i} & \forall \, f \in \mathcal{F}, \ \forall \, i \in \mathcal{M} \label{eq:71b}\tag{b}\\
#     & \sum\limits_{i \in \mathcal{M}} \sum\limits_{g \in \mathcal{F} \cup \{ 0, F+1\} ~:~ a = (g, f) \in \mathcal{A}} x_{a, i} = 1 & \forall \, f \in \mathcal{F} \label{eq:71c}\tag{c}\\
#     & \sum\limits_{f \in \mathcal{F} \cup \{ 0, F+1\} ~:~ a = (0, f) \in \mathcal{A}} x_{a, i} \leq 1 & \forall \, i \in \mathcal{M} \label{eq:71d}\tag{d}\\
#     & x_{(f, 0), i} \leq y_i & \forall \, f \in \mathcal{F}, \ \forall \, i \in \mathcal{M} \label{eq:71e}\tag{e}\\
#     & y_i \leq y_{i - 1} & \forall \, i \in \mathcal{M}: \ i > 0 \label{eq:71f}\tag{f}\\
#     & x_{a, i} \in \{0,1\} & \forall \, a \in \mathcal{A}, i \in \mathcal{M}\\
#     & y_i \in \{0,1\} & \forall \, i \in \mathcal{M} 
# \end{align}
# 
# where
# - the objective function (\ref{eq:71a}) aims to minimize the number of used airplanes;
# - constraint (\ref{eq:71b}) enforces that for a given real flight the number of used incoming arcs with airplane $i$ must be equal to the number of outgoing arcs with this airplane;
# - constraint (\ref{eq:71c}) ensures that  each flight is operated by exactly one aircraft;
# - constraint (\ref{eq:71d}) enforces that each airplane serves at most one path;
# - constraint (\ref{eq:71e}) ensures that if at least one arc $(f, 0)$ is used using a given airplane, then this airplane is used;
# - \eqref{eq:71f} is a symmetry-breaking constraint, which makes sure that we only use airplanes that come first in the aircraft set.
# 

# In[316]:


import networkx as nx

DG = nx.DiGraph()

# make sure all nodes are displayed, even if unconnected
for flight in FlightData.index:
    DG.add_node(flight)


flight_pairs = filter(
    lambda pair: FlightData.loc[pair[0], "Arrival"] < FlightData.loc[pair[1], "Departure"],
    [[i, j] for i in FlightData.index for j in FlightData.index if i != j]
)

# ordered pairs
for flight1, flight2 in flight_pairs:
    DG.add_edge(flight1, flight2)
    if FlightData.loc[flight1, "Arrival"] + 1 == FlightData.loc[flight2, "Departure"]:
        DG[flight1][flight2]["color"] = "r"
    else:
        DG[flight1][flight2]["color"] = "k"
        
# add dummy START
DG.add_node("START")
for flight in FlightData.index:
    DG.add_edge("START", flight)
    DG["START"][flight]["color"] = "b"
    
# add dummy FINISH
DG.add_node("FINISH")
for flight in FlightData.index:
    DG.add_edge(flight, "FINISH")
    DG[flight]["FINISH"]["color"] = "b"
    
fig = plt.figure(figsize=(8, 8))
nx.draw(DG, nx.circular_layout(DG), with_labels=True, font_size=8, arrowsize=10,
        node_color="gold", edge_color=[DG.edges[u, v]["color"] for u, v in DG.edges])


# In[4]:


def flight_network(FlightData):
    
    # create a list of pairs of potentially subsequent flights
    flight_pairs = []

    for flight1 in FlightData.index:
        for flight2 in FlightData.index:
            if FlightData.loc[flight1, "Arrival"] < FlightData.loc[flight2, "Departure"]:
                flight_pairs.append((flight1, flight2))

    nb_flight_pairs = len(flight_pairs)

    pair_riskiness = {pair: 1 if FlightData.loc[pair[0], "Arrival"] + 1 == FlightData.loc[pair[1], "Departure"] else 0 for pair in flight_pairs}

    # create a dictionary of a pair of flights being risky or not
    dummy_node_index = max(FlightData.index) + 1
    start_pairs = []
    end_pairs = []
    
    for flight1 in FlightData.index:
        
        # dummy incoming node
        flight_pairs.append((dummy_node_index, flight1))
        start_pairs.append((dummy_node_index, flight1))
        pair_riskiness[(dummy_node_index, flight1)] = 0
        dummy_node_index += 1

        # dummy outgoing node
        flight_pairs.append((flight1, dummy_node_index))
        end_pairs.append((flight1, dummy_node_index))
        pair_riskiness[(flight1, dummy_node_index)] = 0
        dummy_node_index += 1
        
    return flight_pairs, pair_riskiness, start_pairs, end_pairs

def fleet_assignment(FlightData, flight_pairs, pair_riskiness, start_pairs, end_pairs):
    
    m = pyo.ConcreteModel("Fleet Assignment")
    max_planes = FlightData.shape[0]
    
    m.A = pyo.Set(initialize=list(range(max_planes)))
    m.F = pyo.Set(initialize=FlightData.index)
    m.P = pyo.Set(initialize=pair_riskiness.keys())
    m.x = pyo.Var(m.A, m.P, domain=pyo.Binary)
    m.y = pyo.Var(m.A, domain=pyo.Binary)
          
    # constraint: balance equations for all flights and planes
    @m.Constraint(m.A, m.F)
    def balance(m, aircraft, flight):
        return pyo.quicksum(m.x[aircraft, pair] for pair in pair_riskiness if pair[1] == flight) == pyo.quicksum(m.x[aircraft, pair] for pair in pair_riskiness if pair[0] == flight)

    # each flight assigned to exactly one aircraft
    @m.Constraint(m.F)
    def flightcovered(m, flight):
        return pyo.quicksum(m.x[aircraft, pair] for aircraft in m.A for pair in pair_riskiness if pair[1] == flight) == 1

    # each aircraft operates only one route
    @m.Constraint(m.A)
    def aircraftstarts(m, aircraft):
        return pyo.quicksum(m.x[aircraft, pair] for pair in start_pairs) <= 1
    
    @m.Constraint(m.A)
    def aircraftends(m, aircraft):
        return pyo.quicksum(m.x[aircraft, pair] for pair in end_pairs) <= 1

    # if aircraft serves at least one pair then it is considered used
    @m.Constraint(m.A, m.P)
    def aircraftisused(m, aircraft, pair0, pair1):
        return m.x[aircraft, (pair0, pair1)] <= m.y[aircraft]

    # symmetry breaking constraint
    @m.Constraint(m.A)
    def symmetrybreaking(m, aircraft):
        if aircraft > 0:
            return m.y[aircraft] <= m.y[aircraft - 1]
        else:
            return pyo.Constraint.Skip
    
    @m.Expression()
    def usedplanes(m):
        return pyo.quicksum(m.y[aircraft] for aircraft in m.A)

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return m.usedplanes

    m.solver = pyo.SolverFactory("gurobi")
    m.solver.solve(m)
    
    return m

def visualize(m, FlightData, name, seed = 0):
    
    N_flights = len(FlightData)
    FlightData["Aircraft"] = -1
    FlightData["Aircraft"].astype('int64')

    for aircraft in m.A:
        if m.y[aircraft]() > 0:
            flights = []
            for pair in flight_pairs:
                if m.x[aircraft, pair]() > 0:
                    if pair[0] < N_flights:
                        flights.append(pair[0])
                    if pair[1] < N_flights:
                        flights.append(pair[1])

            for flight in set(flights):
                FlightData.loc[flight, "Aircraft"] = aircraft
            
    JOBS = sorted(list(FlightData.index))
    AIRCRAFTS = sorted(list(FlightData['Aircraft'].unique()))
    makespan = FlightData['Arrival'].max()
    nb_aircraft = FlightData["Aircraft"].max() + 1
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
    fig, ax = plt.subplots(figsize = (FlightData["Arrival"].max(), nb_aircraft / 2))
    
    for flight in FlightData.index:
        xs = FlightData.loc[flight, 'Departure']
        xf = FlightData.loc[flight, 'Arrival']
        jdx = FlightData.loc[flight, "Aircraft"]
        ax.plot([xs, xf], [jdx]*2, c = 'grey', **bar_style)
        ax.text((xs + xf)/2, jdx, str(flight), **text_style)
    
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Plane number")
    ax.set_yticks(range(0, nb_aircraft))
    ax.set_xticks(np.arange(0, 25, 3))
    fig.tight_layout()
    plt.savefig(name + "_seed=" + str(seed) + ".pdf")
    plt.show()
    
    return None

flight_pairs, pair_riskiness, start_pairs, end_pairs = flight_network(FlightData)
m = fleet_assignment(FlightData, flight_pairs, pair_riskiness, start_pairs, end_pairs)
print(f'The optimal solution for this instance of the fleet assignment uses a total of {pyo.value(m.objective):.0f} planes.\n')
total_riskiness = sum([pair_riskiness[pair] * m.x[aircraft, pair]() for aircraft in m.A for pair in pair_riskiness])
print(f'The number of risky pairs (back-to-back flights operated by the same airplane with exactly 1h between them) in this solution is equal to {total_riskiness:.0f}.\n')
visualize(m, FlightData, "optimalschedule", seed)


# ## Reducing riskiness of the schedule
# 
# We will now to keep the maximum number of flights at the optimal level, but try to minimize their riskiness. To do so, we define a slightly different MILP that takes the minimum number of planes `nplanes` in input and has the total number of risky pairs as objective function.
# 

# In[5]:


def fleet_assignment_minimum_risk(FlightData, flight_pairs, pair_riskiness, start_pairs, end_pairs, nplanes):
    
    m = pyo.ConcreteModel("Fleet Assignment with minimum risk")
    max_planes = FlightData.shape[0]
    
    m.A = pyo.Set(initialize=list(range(max_planes)))
    m.F = pyo.Set(initialize=FlightData.index)
    m.P = pyo.Set(initialize=pair_riskiness.keys())
    m.x = pyo.Var(m.A, m.P, domain=pyo.Binary)
    m.y = pyo.Var(m.A, domain=pyo.Binary)
          
    # constraint: balance equations for all flights and planes
    @m.Constraint(m.A, m.F)
    def balance(m, aircraft, flight):
        return pyo.quicksum(m.x[aircraft, pair] for pair in pair_riskiness if pair[1] == flight) == pyo.quicksum(m.x[aircraft, pair] for pair in pair_riskiness if pair[0] == flight)

    # each flight assigned to exactly one aircraft
    @m.Constraint(m.F)
    def flightcovered(m, flight):
        return pyo.quicksum(m.x[aircraft, pair] for aircraft in m.A for pair in pair_riskiness if pair[1] == flight) == 1

    # each aircraft operates only one route
    @m.Constraint(m.A)
    def aircraftstarts(m, aircraft):
        return pyo.quicksum(m.x[aircraft, pair] for pair in start_pairs) <= 1
    
    @m.Constraint(m.A)
    def aircraftends(m, aircraft):
        return pyo.quicksum(m.x[aircraft, pair] for pair in end_pairs) <= 1

    # if aircraft serves at least one pair then it is considered used
    @m.Constraint(m.A, m.P)
    def aircraftisused(m, aircraft, pair0, pair1):
        return m.x[aircraft, (pair0, pair1)] <= m.y[aircraft]

    # symmetry breaking constraint
    @m.Constraint(m.A)
    def symmetrybreaking(m, aircraft):
        if aircraft > 0:
            return m.y[aircraft] <= m.y[aircraft - 1]
        else:
            return pyo.Constraint.Skip
    
    # fixed maximum number of planes
    @m.Constraint()
    def usedplanes(m):
        return pyo.quicksum(m.y[aircraft] for aircraft in m.A) <= nplanes
    
    @m.Expression()
    def riskypairs(m):
        return pyo.quicksum(pair_riskiness[pair] * m.x[aircraft, pair] for aircraft in m.A for pair in pair_riskiness)
    
    # new objective function minimizing riskiness
    @m.Objective(sense=pyo.minimize)
    def riskobjective(m):
        return m.riskypairs
    
    m.solver = pyo.SolverFactory("gurobi")
    m.solver.solve(m)
    
    return m

# reobtain a feasible number of planes by solving the previous model
flight_pairs, pair_riskiness, start_pairs, end_pairs = flight_network(FlightData)
m1 = fleet_assignment(FlightData, flight_pairs, pair_riskiness, start_pairs, end_pairs)
nplanes = pyo.value(m1.objective)

m2 = fleet_assignment_minimum_risk(FlightData, flight_pairs, pair_riskiness, start_pairs, end_pairs, nplanes)
print(f'The optimal solution for this instance of the fleet assignment uses a total of {pyo.value(pyo.quicksum(m2.y[aircraft] for aircraft in m2.A)):.0f} planes.\n')
print(f'The number of risky pairs (back-to-back flights operated by the same airplane with exactly 1h between them) in this new solution is equal to {pyo.value(m2.riskobjective):.0f}.\n')
visualize(m2, FlightData, "optimalschedule_minrisk", seed)

