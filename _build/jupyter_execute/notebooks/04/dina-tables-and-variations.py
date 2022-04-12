#!/usr/bin/env python
# coding: utf-8

# # Dina's table seating arrangements

# In[1]:


# Install Pyomo and solvers for Google Colab
import sys
if "google.colab" in sys.modules:
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Problem
# 
# Caroline takes her associates and their families out to a dinner at Dina’s restaurant to celebrate the successes brought to her material planning by Mathematical Optimization.
# 
# 
# Caroline puts some additional requirements: to increase social interaction, the different families should sit at tables so that no more than $k$ members of the same family are seated at the same table.
# 
# Dina has the following data, plus the desired threshold of at most $k$ family members per table:
# 
#  * Family $f$ has $m(f)$ members. 
#  * At the restaurant, there are multiple tables, where table $t$ has capacity $c(t)$.
# 
# Model this problem in order to find a seating arrangement that satisfies Caroline's requirement.

# # Resolution

# In[2]:


import pyomo.environ as pyo
from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


solver = pyo.SolverFactory('cbc')


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


seatplan = TableSeat( [6,8,2,9,13,1], [8,8,10,4,9], 3 )

get_ipython().run_line_magic('time', 'results = solver.solve(seatplan)')

Report( seatplan,results )


# In[8]:


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

# In[9]:



get_ipython().run_line_magic('matplotlib', 'inline')

# https://stackoverflow.com/questions/17687213/how-to-obtain-the-same-font-style-size-etc-in-matplotlib-output-as-in-latex
params = {'text.usetex' : True,
          'font.size'   : 10, # the book seems to be in 10pt, change if needed
          'font.family' : 'lmodern',
          }

plt.rcParams.update(params)
default_size_inches = (3.54,3.54) 
plt.rcParams['figure.figsize'] = default_size_inches


# In[10]:


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


# In[11]:


G = ModelAsNetwork( [6,8,2,9,13,1], [8,8,10,4,9], 3 )


# In[12]:


pos = nx.multipartite_layout(G, subset_key='layer')


# In[13]:


labels = { (e[0],e[1]) : e[2] for e in G.edges(data='capacity') }


# In[14]:


plt.rcParams['text.usetex'] =False
with plt.xkcd():
    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111)
    nx.draw_networkx(G,pos=pos,ax=ax,node_size=800,with_labels=True,alpha=.6)
    _=nx.draw_networkx_edge_labels(G,pos=pos,ax=ax,edge_labels=labels,font_color='black',rotate=False,alpha=1)
    fig.savefig( 'net_flow.pdf', bbox_inches='tight', pad_inches=0 )


# In[15]:


get_ipython().run_line_magic('time', "flow_value, flow_dict = nx.maximum_flow(G, 'door', 'seat')")


# In[16]:


members, capacity = [6,8,2,9,13,1], [8,8,10,4,9]
families = [f'f{i}' for i in range(len(members))]
tables = [f't{j}' for j in range(len(capacity))]
pd.DataFrame(flow_dict).loc[tables,families]


# In[17]:


flow_edges = [(a,b) for a,B in flow_dict.items() for b,v in B.items() if v>0 and a != 'door' and b != 'seat']
flow_nodes = [n for n in G.nodes if n.startswith('f') or n.startswith('t')]


# In[18]:


with plt.xkcd():
    fig = plt.figure(figsize=(8,5))
    nx.draw_networkx(G,ax=fig.add_subplot(111),pos=pos,node_size=300,edge_color='blue',edgelist=flow_edges,nodelist=flow_nodes)


# In[19]:


fig.savefig( 'flow.pdf', bbox_inches='tight', pad_inches=0 )


# In[20]:


cbc    = pyo.SolverFactory('cbc')
gurobi = pyo.SolverFactory('gurobi_direct')


# In[21]:


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


# In[22]:


aux = df.T


# In[23]:


df


# In[24]:


aux


# In[25]:


import numpy as np
plt.rcParams['text.usetex'] =True
fig = plt.figure(figsize=(13,5))
ax=fig.add_subplot(111)
aux.plot(ax=ax)
plt.xticks(np.arange(len(df.columns)),df.columns,rotation = 45)
plt.show()


# In[26]:


fig.savefig( 'dina_times.pdf', bbox_inches='tight', pad_inches=0 )


# In[ ]:





# In[ ]:




