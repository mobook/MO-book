#!/usr/bin/env python
# coding: utf-8

# ```{index} single: application; seating allocation
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} single: Pyomo; parameters
# ```
# ```{index} single: Pyomo; sets
# ```
# ```{index} network optimization
# ```
# ```{index} max flow problem
# ```
# ```{index} networkx
# ```
# ```{index} feasibility problem
# ```
# 
# # Dinner seating arrangement

# ## Preamble: Install Pyomo and a solver
# 
# This cell selects and verifies a global SOLVER for the notebook.
# 
# If run on Google Colab, the cell installs Pyomo and HiGHS, then sets SOLVER to 
# use the Highs solver via the appsi module. If run elsewhere, it assumes Pyomo and CBC
# have been previously installed and sets SOLVER to use the CBC solver via the Pyomo 
# SolverFactory. It then verifies that SOLVER is available.

# In[1]:


import sys

if 'google.colab' in sys.modules:
    get_ipython().system('pip install pyomo >/dev/null 2>/dev/null')
    get_ipython().system('pip install highspy >/dev/null 2>/dev/null')

    from pyomo.contrib import appsi
    SOLVER = appsi.solvers.Highs(only_child_vars=False)
    
else:
    from pyomo.environ import SolverFactory
    SOLVER = SolverFactory('cbc')

assert SOLVER.available(), f"Solver {SOLVER} is not available."


# ## Problem description
# 
# Assume that you are organizing a wedding dinner at which your objective is to have guests from different families mingle with each other. One way to do this is to seat people at tables so that no more people than a given threshold $k$ from the same family sit at the same table. How could we solve a problem like this? 
# 
# First, we need the problem data -- for each family $f$ we need to know the number of its members $m_f$, and for each table $t$ we need to know its capacity $c_t$. Using these data and the tools we have learned so far, we can formulate this problem as a LO problem.
# 
# We can use variable $x_{ft}$  for the number of persons from family $f$ to be seated at table $t$. Since we were not provided with any objective function, we can focus on finding a feasible solution by setting the objective function to be constant, say $0$, which means that we do not differentiate between feasible solutions. 
# 
# The mathematical formulation of this seating problem is:
# 
# $$
# \begin{align*}
#     \min_{x_{ft}} \quad & 0\\
#     \text{s.t.} \quad & \sum\limits_{f} x_{ft} \leq c_t & \forall \, t \in T \\
#     & \sum\limits_{t} x_{ft} = m_f & \forall \, f \in F \\
#     & 0 \leq x_{ft} \leq k.
# \end{align*}
# $$
# 
# The constraints ensure that the seating capacity for each table is not exceeded, that each family is fully seated, and that the number of elements of each family at each table does not exceed the threshold $k$.

# ## Implementation
# 
# The problem statement will be satisfied by finding a feasible solution, if one exists. Because no specific objective has been specified, the mathematical formulation uses a constant 0 as essentially a placeholder for the objective function. Some optimization solvers, however, issue warning messages if they detect a constant objective, and others will fail to execute at all. A simple work around for these cases is to replace the constant objective with a 'dummy' variable that doesn't  appear elsewhere in the optimization problem. 

# In[37]:


import pyomo.environ as pyo
from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def table_seat(members, capacity, k, domain=pyo.NonNegativeReals):
    
    m = pyo.ConcreteModel("Dina's seat plan")
    
    m.F = pyo.Set(initialize=range(len(members)))
    m.T = pyo.Set(initialize=range(len(capacity)))
    
    m.M = pyo.Param(m.F, initialize=members)
    m.C = pyo.Param(m.T, initialize=capacity)
    
    m.x = pyo.Var(m.F, m.T, bounds=(0, k), domain=domain)
    m.dummy = pyo.Var(bounds=(0, 1), initialize=0)
    
    @m.Objective(sense=pyo.minimize)
    def goal(m):
        return m.dummy

    @m.Constraint(m.T)    
    def capacity(m, t):
        return pyo.quicksum(m.x[f, t] for f in m.F) <= m.C[t]
    
    @m.Constraint(m.F)
    def seat(m, f):
        return pyo.quicksum(m.x[f, t] for t in m.T) == m.M[f]
        
    return m

def get_solution(model):
    df = pd.DataFrame()
    for idx, x in model.x.items():
        f, t = idx
        df.loc[f, t] = x()
    df.index.name = "family"
    df.columns = [f"table {i}" for i in model.T]
    return df.round(5)
    
def report(model, results, type=int):
    print(f"Solver status: {results.solver.status}")
    print(f"Termination condition: {results.solver.termination_condition}")
    if results.solver.termination_condition == 'optimal':
        soln = get_solution(model).astype(type)
        display(soln)
        print(f'objective:       {pyo.value(seatplan.goal)}')
        print(f'places at table: {list(soln.sum(axis=0))}')
        print(f'members seated:  {list(soln.sum(axis=1))}')


# Let us now consider and solve a specific instance of this problem with six families with sizes $m = (6, 8, 2, 9, 13, 1)$, five tables with capacities $c = (8, 8, 10, 4, 9)$, and a threshold $k=3$ for the number of members of each family that can be seated at the same table. 

# In[38]:


seatplan = table_seat(
    members=[6, 8, 2, 9, 13, 1], 
    capacity=[8, 8, 10, 4, 9], 
    k=3
)
get_ipython().run_line_magic('time', 'results = SOLVER.solve(seatplan)')
report(seatplan, results, type=float)


# A peculiar fact is that although we did not explicitly require that all variables $x_{ft}$ be integer, the optimal solution turned out to be integer anyway. This is no coincidence as it follows from a certain property of the problem we solve. This also means we can solve  larger versions of this problem with LO instead of MILO solvers to find integer solutions, gaining a large computational advantage.

# ## Minimize the maximum group size
# 
# Our objective was that we make members of different families mingle as much as possible. Is $k = 3$ the lowest possible number for which a feasible table allocation exists or can we make the tables even more diverse by bringing this number down?
# 
# In order to find out, we change the objective function and try to minimize $k$, obtaining the following problem:

# In[39]:


def table_seat_minimize_max_group_at_table(members, capacity, domain=pyo.NonNegativeReals):
    
    m   = pyo.ConcreteModel("Dina's seat plan")
    
    m.F = pyo.Set(initialize=range(len(members)))
    m.T = pyo.Set(initialize=range(len(capacity)))
    
    m.M = pyo.Param(m.F, initialize=members)
    m.C = pyo.Param(m.T, initialize=capacity)
    
    m.x = pyo.Var(m.F, m.T, domain=domain)
    m.k = pyo.Var( domain=domain)
    
    @m.Objective(sense=pyo.minimize)
    def goal(m):
        return m.k
    
    @m.Constraint(m.T)    
    def capacity(m, t):
        return pyo.quicksum(m.x[f,t] for f in m.F  ) <= m.C[t]
    
    @m.Constraint(m.F)
    def seat(m, f):
        return pyo.quicksum(m.x[f,t] for t in m.T ) == m.M[f]

    @m.Constraint(m.F, m.T)
    def bound(m, f, t):
        return m.x[f,t] <= m.k

    return m


# We now solve the same instance as before.

# In[40]:


seatplan = table_seat_minimize_max_group_at_table(
    members=[6, 8, 2, 9, 13, 1], 
    capacity=[8, 8, 10, 4, 9], 
    domain=pyo.NonNegativeReals 
)
get_ipython().run_line_magic('time', 'results = SOLVER.solve(seatplan)')
report(seatplan, results, type=float)


# Unfortunately, this solution is no longer integer. Mathematically, this is because the "structure" that previously ensured integer solutions at no extra cost has been lost as a result of making $k$ a decision variable. To find the solution to this problem we need to impose that the variables are integers.
# 
# Using an MILO solver such as `cbc` or `highs`, we recover the original optimal value $k = 3$.

# In[41]:


seatplan = table_seat_minimize_max_group_at_table(
    members=[6, 8, 2, 9, 13, 1], 
    capacity=[8, 8, 10, 4, 9], 
    domain=pyo.NonNegativeIntegers
)
get_ipython().run_line_magic('time', 'results = SOLVER.solve(seatplan)')
report(seatplan, results, type=int)


# ## Minimize number of tables

# In[42]:


def table_seat_minimize_number_of_tables(members, capacity, k, domain=pyo.NonNegativeReals):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set(initialize=range(len(members)))
    m.T = pyo.Set(initialize=range(len(capacity)))
    m.M = pyo.Param(m.F, initialize=members)
    m.C = pyo.Param(m.T, initialize=capacity)
    m.x = pyo.Var(m.F, m.T, bounds=(0,k), domain=domain)
    m.y = pyo.Var(m.T, within=pyo.Binary)
    
    @m.Objective(sense=pyo.minimize)
    def goal(m):
        return pyo.quicksum(m.y[t] for t in m.T)
    
    @m.Constraint(m.T)    
    def capacity(m, t):
        return pyo.quicksum(m.x[f,t] for f in m.F  ) <= m.C[t]*m.y[t]
    
    @m.Constraint(m.F)
    def seat(m, f):
        return pyo.quicksum(m.x[f,t] for t in m.T) == m.M[f]

    return m


# In[43]:


seatplan = table_seat_minimize_number_of_tables(
    members=[6, 8, 2, 9, 13, 1], 
    capacity=[8, 8, 10, 4, 9], 
    k=3, 
    domain=pyo.NonNegativeIntegers
)
get_ipython().run_line_magic('time', 'results = SOLVER.solve(seatplan)')
report(seatplan, results, type=int)


# # Reformulation as max flow problem

# However, using an MILO solver is not necessarily the best approach for problems like this. Many real-life situations (assigning people to work/course groups) require solving really large problems.  There are existing algorithms that can leverage the special **network structure** of the problem at hand and scale better than LO solvers. To see this we can visualize the seating problem using a graph where:
# 
# * the nodes on the left-hand side stand for the families and the numbers next to them provide the family size
# * the nodes on the left-hand side stand for the tables and the numbers next to them provide the table size
# * each left-to-right arrow stand comes with a number denoting the capacity of arc $(f, t)$ -- how many people of family $f$ can be assigned to table $t$.
# 
# ![](dina_model_basic.png)

# If we see each family as a place of supply (people) and tables as places of demand (people), then we can see our original problem as literally sending people from families $f$ to tables $t$ so that everyone is assigned to some table, the tables' capacities are respected, and no table gets more than $k = 3$ members of the same family.
# 
# A Pyomo version of this model is given in the next cell. After that we will show how to reformulate the calculation using network algorithms.

# In[44]:


def table_seat_maximize_members_flow_to_tables(members, capacity, k, domain=pyo.NonNegativeReals):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set(initialize=range(len(members)))
    m.T = pyo.Set(initialize=range(len(capacity)))
    m.M = pyo.Param(m.F, initialize=members)
    m.C = pyo.Param( m.T, initialize=capacity)
    m.x = pyo.Var(m.F, m.T, bounds=(0, k), domain=domain)
    
    @m.Objective(sense=pyo.maximize)
    def goal(m):
        return pyo.quicksum(m.x[f, t] for f in m.F for t in m.T)
    
    @m.Constraint(m.T)    
    def capacity(m, t):
        return pyo.quicksum(m.x[f,t] for f in m.F  ) <= m.C[t]
    
    @m.Constraint(m.F)
    def seat(m, f):
        return pyo.quicksum(m.x[f,t] for t in m.T ) == m.M[f]

    return m

seatplan = table_seat_maximize_members_flow_to_tables(
    members=[6, 8, 2, 9, 13, 1], 
    capacity=[8, 8, 10, 4, 9], 
    k=3, 
    domain=pyo.NonNegativeIntegers
)
get_ipython().run_line_magic('time', 'results = SOLVER.solve(seatplan)')
report(seatplan, results, type=int)


# By adding two more nodes to the graph above, we can formulate the problem as a slightly different flow problem where all the data is formulated as the arc capacity, see figure below. In a network like this, we can imagine a problem of sending resources from the _root node_ "door" to the _sink node_ "seat", subject to the restriction that for any node apart from $s$ and $t$, the sum of incoming and outgoing flows are equal (_balance constraint_). If there exists a flow in this new graph that respects the arc capacities and the sum of outgoing flows at $s$ is equal to $\sum_{f \in F} m_f = 39$, it means that there exists a family-to-table assignment that meets our requirements.
# 
# ![](dina_model.png)

# In[9]:


def model_as_network(members, capacity, k):
    
    # create lists of families and tables
    families = [f'f{i}' for i in range(len(members))]
    tables = [f't{j}' for j in range(len(capacity))]
    
    # create digraphy object
    G = nx.DiGraph()
    
    # add edges
    G.add_edges_from(['door', f, {'capacity': n}] for f, n in zip(families, members))
    G.add_edges_from([(f, t) for f in families for t in tables], capacity=k)
    G.add_edges_from([t, 'seat', {'capacity': n}] for t, n in zip(tables, capacity))

    return G


# In[10]:


members = [6, 8, 2, 9, 13, 1]
capacity = [8, 8, 10, 4, 9]

G = model_as_network(members, capacity=[8, 8, 10, 4, 9], k=3)

labels = {(e[0], e[1]) : e[2] for e in G.edges(data='capacity')}


# In[11]:


get_ipython().run_line_magic('time', "flow_value, flow_dict = nx.maximum_flow(G, 'door', 'seat')")


# In[52]:


families = [f'f{i:.0f}' for i in range(len(members))]
tables = [f't{j:.0f}' for j in range(len(capacity))]

pd.DataFrame(flow_dict).loc[tables, families].astype('int')


# Even for this very small example, we see that network algorithms generate a solution significantly faster.

# In[ ]:




