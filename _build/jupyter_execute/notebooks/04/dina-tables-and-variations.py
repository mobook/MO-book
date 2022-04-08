#!/usr/bin/env python
# coding: utf-8

# # Dina's table seating arrangements
# 
# ---
# Caroline takes her associates and their families out to a dinner at Dina’s restaurant to celebrate the successes brought to her material planning by Mathematical Optimization.
# 
# 
# <img width=650 src='https://img.freepik.com/free-vector/cheerful-sexy-girl-restaurant-waiter-with-tray-wine-glass-portrait-isolated-white-background-vector-illustration_1284-2391.jpg?w=826&t=st=1649165412~exp=1649166012~hmac=6d1567c8ce54bac51bdb09a39111f53df051eb5983c0f0edc1f5ddb2ae641d67'>
# 
# <a href="https://www.freepik.com/vectors/background-people">Background people vector created by macrovector - www.freepik.com</a>
# 
# Caroline puts some additional requirements: to increase social interaction, the different families should sit at tables so that no more than $k$ members of the same family are seated at the same table.
# 
# Dina has the following data, plus the desired threshold of at most $k$ family members per table:
# 
#  * Family $f$ has $m(f)$ members. 
#  * At the restaurant, there are multiple tables, where table $t$ has capacity $c(t)$.
# 
# Model this problem in order to find a seating arrangement that satisfies Caroline's requirement.
# 
# ---

# # Resolution

# In[ ]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# In[ ]:


import pyomo.environ as pyo
from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


solver = pyo.SolverFactory('cbc')


# In[ ]:


# %%writefile tableseat_1.py
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


# In[ ]:


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


# In[ ]:


def Reset( model ) -> None:
    for v in model.component_data_objects(ctype=pyo.Var, descend_into=True):
        v.set_value(None)
        
def GetSolution( model ):
    import pandas as pd
    sol = pd.DataFrame()
    for idx,x in model.x.items():
        sol.loc[idx]=x()
    return sol
    
def Report( model, results, type=float ):
    print(results.solver.status, results.solver.termination_condition )
    if results.solver.termination_condition == 'optimal':
        sol = GetSolution(model).astype(type)
        display( sol )
        print('objective       ', pyo.value( seatplan.goal ) )
        print('places at table ', list(sol.sum(axis=0)))
        print('members seated  ', list(sol.sum(axis=1)))


# In[ ]:


seatplan = TableSeat( [6,8,2,9,13,1], [8,8,10,4,9], 3 )

get_ipython().run_line_magic('time', 'results = solver.solve(seatplan)')

Report( seatplan,results )


# In[ ]:


import pyperclip 
pyperclip.copy( GetSolution(seatplan).astype(int).style.to_latex() )


# In[ ]:


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


# In[ ]:


seatplan = TableSeatMinimizeMaxGroupAtTable( [6,8,2,9,13,1], [8,8,10,4,9], nature=pyo.NonNegativeReals )

get_ipython().run_line_magic('time', 'results = solver.solve(seatplan)')

Report( seatplan, results )


# In[ ]:


pyperclip.copy( GetSolution(seatplan).astype(float).style.format(precision=2).to_latex() )


# In[ ]:


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


# In[ ]:


seatplan = TableSeatMinimizeNumberOfTables( [6,8,2,9,13,1], [8,8,10,4,9], 3, pyo.NonNegativeIntegers )

get_ipython().run_line_magic('time', 'results = solver.solve(seatplan)')

Report( seatplan, results, int )


# # Note: this is an example of a max flow!

# In[ ]:



get_ipython().run_line_magic('matplotlib', 'inline')

# https://stackoverflow.com/questions/17687213/how-to-obtain-the-same-font-style-size-etc-in-matplotlib-output-as-in-latex
params = {'text.usetex' : True,
          'font.size'   : 10, # the book seems to be in 10pt, change if needed
          'font.family' : 'lmodern',
          }

plt.rcParams.update(params)
default_size_inches = (3.54,3.54) 
plt.rcParams['figure.figsize'] = default_size_inches


# In[ ]:


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


# In[ ]:


G = ModelAsNetwork( [6,8,2,9,13,1], [8,8,10,4,9], 3 )


# In[ ]:


pos = nx.multipartite_layout(G, subset_key='layer')


# In[ ]:


labels = { (e[0],e[1]) : e[2] for e in G.edges(data='capacity') }


# In[ ]:


plt.rcParams['text.usetex'] =False
with plt.xkcd():
    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111)
    nx.draw_networkx(G,pos=pos,ax=ax,node_size=800,with_labels=True,alpha=.6)
    _=nx.draw_networkx_edge_labels(G,pos=pos,ax=ax,edge_labels=labels,font_color='black',rotate=False,alpha=1)
    fig.savefig( 'net_flow.pdf', bbox_inches='tight', pad_inches=0 )


# In[ ]:


get_ipython().run_line_magic('time', "flow_value, flow_dict = nx.maximum_flow(G, 'door', 'seat')")


# In[ ]:


members, capacity = [6,8,2,9,13,1], [8,8,10,4,9]
families = [f'f{i}' for i in range(len(members))]
tables = [f't{j}' for j in range(len(capacity))]
pd.DataFrame(flow_dict).loc[tables,families]


# In[ ]:


flow_edges = [(a,b) for a,B in flow_dict.items() for b,v in B.items() if v>0 and a != 'door' and b != 'seat']
flow_nodes = [n for n in G.nodes if n.startswith('f') or n.startswith('t')]


# In[ ]:


with plt.xkcd():
    fig = plt.figure(figsize=(8,5))
    nx.draw_networkx(G,ax=fig.add_subplot(111),pos=pos,node_size=300,edge_color='blue',edgelist=flow_edges,nodelist=flow_nodes)


# In[ ]:


fig.savefig( 'flow.pdf', bbox_inches='tight', pad_inches=0 )


# In[ ]:


cbc    = pyo.SolverFactory('cbc')
gurobi = pyo.SolverFactory('gurobi_direct')


# In[ ]:


from pathlib import Path
if Path('dina_times.xlsx').is_file():
    df = pd.read_excel('dina_times.xlsx').set_index('Unnamed: 0')
else:
    from tqdm.notebook import tqdm
    from time import perf_counter as pc
    import numpy as np
    np.random.seed(2022)
    k = 3
    nmax = 500
    mmax = 2*nmax
    sizes = list(zip(range(10,nmax,10),range(20,mmax,20)))
    df = pd.DataFrame(index=['cbc','gurobi','nx'],columns=sizes)
    for n,m in tqdm(sizes):
        members, capacity = np.random.randint(1,10,n), np.random.randint(3,8,m)
        model = TableSeatAsMaxFlow(members,capacity,k)
        t=pc() 
        cbc.solve(model)
        df.loc['cbc'][(n,m)] = pc()-t
        Reset(model)
        t=pc() 
        gurobi.solve(model)
        df.loc['gurobi'][(n,m)] = pc()-t
        G = ModelAsNetwork(members,capacity,k)
        t = pc()
        nx.maximum_flow(G, 'door', 'seat')
        df.loc['nx'][(n,m)] = pc()-t
        
    df.to_excel('dina_times.xlsx')


# In[ ]:


aux = df.T


# In[ ]:


df


# In[ ]:


aux


# In[ ]:


import numpy as np
plt.rcParams['text.usetex'] =True
fig = plt.figure(figsize=(13,5))
ax=fig.add_subplot(111)
aux.plot(ax=ax)
plt.xticks(np.arange(len(df.columns)),df.columns,rotation = 45)
plt.show()


# In[ ]:


fig.savefig( 'dina_times.pdf', bbox_inches='tight', pad_inches=0 )


# In[ ]:




