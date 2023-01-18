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
# 
# # Dinner seating arrangement

# In[2]:


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
# Consider organizing a wedding dinner at which your objective is that the guests from different families mingle with each other. One of the ways to do so is to seat people at the tables in such a way that no more people than a given threshold $k$ from the same family take a seat at the same table. How could we solve a problem like this? First, we need the problem data -- for each family $f$ we need to know the number of its members $m_f$, and for each table $t$ we need to know its capacity $c_t$. Using this data and the tools we learned so far, we can formulate this problem as an LP.
# 
# If we do not care about the specific people, but only about the number of people from a given family, then we can use variable $x_{ft}$  for the number of persons from family $f$ to be seated at table $t$. Since we were not provided with any objective function, we can focus on finding a feasible solution by setting the objective function to be constant, say $0$, which means that we do not differentiate between the various feasible solutions. 
# 
# The mathematical formulation of this seating problem is:
# 
# $$
# \begin{align*}
#     \min_{x_{ft}} \quad & 0 \label{ch4eq.Dina.problem.1}\\
#     \text{s.t.} \quad & \sum\limits_{f} x_{ft} \leq c_t & \forall \, t \in T \\
#     & \sum\limits_{t} x_{ft} = m_f & \forall \, f \in F \\
#     & 0 \leq x_{ft} \leq k.
# \end{align*}
# $$
# 
# The two constraints ensure that (i) for each table the seating capacity is not exceeded and that (ii) each family is fully seated and that the number of elements of each family at each table does not exceed $k$.

# ## Implementation

# In[3]:


import pyomo.environ as pyo
from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def TableSeat( members, capacity, k, domain=pyo.NonNegativeReals ):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set( initialize=range( len(members) ) )
    m.T = pyo.Set( initialize=range( len(capacity) ) )
    m.M = pyo.Param( m.F, initialize=members )
    m.C = pyo.Param( m.T, initialize=capacity )
    m.x = pyo.Var( m.F, m.T, bounds=(0,k), domain=domain )
    
    @m.Objective( sense=pyo.maximize )
    def goal(m):
        return 0

    @m.Constraint( m.T )    
    def capacity( m, t ):
        return pyo.quicksum( m.x[f,t] for f in m.F  ) <= m.C[t]
    
    @m.Constraint( m.F )
    def seat( m, f ):
        return pyo.quicksum( m.x[f,t] for t in m.T ) == m.M[f]
        
    return m

def TableSeatAsMaxFlow( members, capacity, k, domain=pyo.NonNegativeReals ):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set( initialize=range( len(members) ) )
    m.T = pyo.Set( initialize=range( len(capacity) ) )
    m.M = pyo.Param( m.F, initialize=members )
    m.C = pyo.Param( m.T, initialize=capacity )
    m.x = pyo.Var( m.F, m.T, bounds=(0,k), domain=domain )
    
    @m.Objective( sense=pyo.maximize )
    def goal(m):
        return pyo.quicksum( m.x[f,t] for t in m.T for f in m.F )

    @m.Constraint( m.T )    
    def capacity( m, t ):
        return pyo.quicksum( m.x[f,t] for f in m.F  ) <= m.C[t]
    
    @m.Constraint( m.F )
    def seat( m, f ):
        return pyo.quicksum( m.x[f,t] for t in m.T ) <= m.M[f]
        
    return m

def Reset( model ) -> None:
    for v in model.component_data_objects(ctype=pyo.Var, descend_into=True):
        v.set_value(None)
        
def GetSolution( model ):
    import pandas as pd
    sol = pd.DataFrame()
    for idx,x in model.x.items():
        sol.loc[idx]=x()
    return sol
    
def Report( model, results, type=int ):
    solver = pyo.SolverFactory('cbc')
    print(results.solver.status, results.solver.termination_condition )
    if results.solver.termination_condition == 'optimal':
        sol = GetSolution(model).astype(type)
        display( sol )
        print('objective       ', pyo.value( seatplan.goal ) )
        print('places at table ', list(sol.sum(axis=0)))
        print('members seated  ', list(sol.sum(axis=1)))


# Let us now consider and solve a specific instance of this problem with family sizes $m = (6,8,2,9,13,1)$, table capacities $c = (8,8,10,4,9)$, and threshold $k=3$. 

# In[4]:


seatplan = TableSeat( [6,8,2,9,13,1], [8,8,10,4,9], 3 )

get_ipython().run_line_magic('time', "results = pyo.SolverFactory('cbc').solve(seatplan)")

Report( seatplan, results )


# A peculiar fact is that although we did not explicitly require that all variables $x_{ft}$ be integer, the optimal solution turned out to be integer anyway. This is no coincidence as it follows from a certain property of the problem we solve. This means that also for larger versions of this problem, we can solve them with LP instead of MILP solvers to find integer solutions, gaining a huge computational advantage.
# 
# Our objective was that we make members of different families mingle as much as possible. Is $k = 3$ the lowest possible number for which a feasible table allocation exists or can we make the tables even more diverse by bringing this number down?
# 
# In order to find out, we change the objective function and try to minimize $k$, obtaining the following problem:

# In[5]:


def TableSeatMinimizeMaxGroupAtTable( members, capacity, nature=pyo.NonNegativeReals ):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set( initialize=range( len(members) ) )
    m.T = pyo.Set( initialize=range( len(capacity) ) )
    m.M = pyo.Param( m.F, initialize=members )
    m.C = pyo.Param( m.T, initialize=capacity )
    m.x = pyo.Var( m.F, m.T, domain=nature )
    m.k = pyo.Var( domain=nature )
    
    @m.Objective( sense=pyo.minimize )
    def goal(m):
        return m.k
    
    @m.Constraint( m.T )    
    def capacity( m, t ):
        return pyo.quicksum( m.x[f,t] for f in m.F  ) <= m.C[t]
    
    @m.Constraint( m.F )
    def seat( m, f ):
        return pyo.quicksum( m.x[f,t] for t in m.T ) == m.M[f]

    @m.Constraint( m.F, m.T )
    def bound( m, f, t ):
        return m.x[f,t] <= m.k

    return m


# We now solve the same instance as before.

# In[6]:


seatplan = TableSeatMinimizeMaxGroupAtTable( [6,8,2,9,13,1], [8,8,10,4,9], nature=pyo.NonNegativeReals )

get_ipython().run_line_magic('time', "results = pyo.SolverFactory('cbc').solve(seatplan)")

Report( seatplan, results, type=float )


# Unfortunately, this solution is no longer integer. Mathematically, this is due to the fact that the "structure" that previously ensured integrality of solutions at no extra cost, has been lost as a result of making $k$ a decision variable. To find the solution to this problem we need to impose that the variables are integers.

# In[7]:


def TableSeatMinimizeNumberOfTables( members, capacity, k, nature=pyo.NonNegativeReals ):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set( initialize=range( len(members) ) )
    m.T = pyo.Set( initialize=range( len(capacity) ) )
    m.M = pyo.Param( m.F, initialize=members )
    m.C = pyo.Param( m.T, initialize=capacity )
    m.x = pyo.Var( m.F, m.T, bounds=(0,k), domain=nature )
    m.y = pyo.Var( m.T, within=pyo.Binary )
    
    @m.Objective( sense=pyo.minimize )
    def goal(m):
        return pyo.quicksum(m.y[t] for t in m.T)
    
    @m.Constraint( m.T )    
    def capacity( m, t ):
        return pyo.quicksum( m.x[f,t] for f in m.F  ) <= m.C[t]*m.y[t]
    
    @m.Constraint( m.F )
    def seat( m, f ):
        return pyo.quicksum( m.x[f,t] for t in m.T ) == m.M[f]

    return m


# Using a MILP solver such as `cbc`, we recover the original optimal value $k = 3$.

# In[8]:


seatplan = TableSeatMinimizeNumberOfTables( [6,8,2,9,13,1], [8,8,10,4,9], 3, pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', "results = pyo.SolverFactory('cbc').solve(seatplan)")

Report( seatplan, results, type=int )


# # Reformulation as max flow problem

# However, using a MILP solver is not necessarily the best approach for problems like this. Many real-life situations (assigning people to work/course groups) require solving really large problems.  There exist algorithms that can leverage the special **network structure** of the problem at hand and which work better than LP solvers. To see this we can visualize the seating problem using a graph where:
# 
# * the nodes on the left-hand side stand for the families and the numbers next to them provide the family size
# * the nodes on the left-hand side stand for the tables and the numbers next to them provide the table size
# * each left-to-right arrow stand comes with a number denoting the capacity of arc $(f, t)$ -- how many people of family $f$ can be assigned to table $t$.
# 
# ![](dina_model_basic.png)

# If we see each family as a place of supply (people) and tables as places of demand (people), then we can see our original problem as literally sending people from families $f$ to tables $t$ so that everyone is assigned to some table, the tables' capacities are respected, and no table gets more than $k = 3$ members of the same family.
# 
# By adding two more nodes to the graph above, we can formulate the problem as a slightly different flow problem where all the data is formulated as the arc capacity, see figure beloow. In a network like this, we can imagine a problem of sending resources from the _root node_ "door" to the _sink node_ "seat", subject to the restriction that for any node apart from $s$ and $t$, the sum of incoming and outgoing flows are equal (_balance constraint_). If there exists a flow in this new graph that respects the arc capacities and the sum of outgoing flows at $s$ is equal to $\sum_{f \in F} m_f = 39$, it means that there exists a family-to-table assignment that meets our requirements.
# 
# ![](dina_model.png)

# In[9]:


def ModelAsNetwork( members, capacity, k ):
    families = [f'f{i}' for i in range(len(members))]
    tables = [f't{j}' for j in range(len(capacity))]
    G = nx.DiGraph()
    G.add_node('door',layer=0)
    for f in families:
        G.add_node(f,layer=1)
    for t in tables:
        G.add_node(t,layer=2)
    G.add_node('seat',layer=3)
    for f,n in zip(families,members):
        G.add_edge('door', f, capacity=n)
    for f in families:
        for t in tables:
            G.add_edge(f,t, capacity=k)
    for t,n in zip(tables,capacity):
        G.add_edge(t, 'seat', capacity=n)
    return G


# In[10]:


G = ModelAsNetwork( [6,8,2,9,13,1], [8,8,10,4,9], 3 )
labels = { (e[0],e[1]) : e[2] for e in G.edges(data='capacity') }


# In[11]:


get_ipython().run_line_magic('time', "flow_value, flow_dict = nx.maximum_flow(G, 'door', 'seat')")


# In[12]:


members, capacity = [6,8,2,9,13,1], [8,8,10,4,9]
families = [f'f{i:.0f}' for i in range(len(members))]
tables = [f't{j:.0f}' for j in range(len(capacity))]
pd.DataFrame(flow_dict).loc[tables,families].astype('int')

