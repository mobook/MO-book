#!/usr/bin/env python
# coding: utf-8

# ```{index} single: application; shortest path
# ```
# ```{index} networkx
# ```
# ```{index} network optimization
# ```
# ```{index} pandas dataframe
# ```
# 
# # Extra material: Shortest path in real life
# 

# In[1]:


import subprocess
import sys

class ColabInstaller():

    def __init__(self):
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        self.installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    def on_colab(self):
        return "google.colab" in sys.modules

    def install(self, package):
        if self.on_colab():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            self.installed_packages.append(package)

    def upgrade(self, package):
        if self.on_colab():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", package])

colab = ColabInstaller()
colab.upgrade('numpy')
colab.upgrade('matplotlib')
colab.install('geopandas')
colab.install('geopy')
colab.install('osmnx')
colab.install('osmnet')
colab.install('pandana')
colab.upgrade('geopy')


# # Introduction
# 
# Google brought with [maps]( https://www.google.com/maps) the world to our screens, including accurate geocoding and routing for several modalities. For the most, the usage of [maps]( https://www.google.com/maps) is interactive. As data and analytics professionals we often need a programmatically support for the services that [maps]( https://www.google.com/maps) offer us, preferably free. It also offers a plethora of [development support](https://developers.google.com/), but unfortunately most is paid. That is even more so for [maps]( https://developers.google.com/maps/documentation).

# ## Some background information and history 
# [Geoff Boeing]( https://geoffboeing.com/about/) is a true leader in demystifying urban data analytics, with a strong emphasis on street networks. His [peer reviewed publications]( https://geoffboeing.com/publications/) are open and accompanied by usable demonstrations using his own [OSMnx]( https://geoffboeing.com/2018/03/osmnx-features-roundup/) package.
# Professor [Peter Sanders]( https://algo2.iti.kit.edu/english/sanders.php), see also his [Wikipedia]( https://en.wikipedia.org/wiki/Peter_Sanders_(computer_scientist)) page, has moved his interests to other areas but his [route planning]( http://algo2.iti.kit.edu/routeplanning.php) project shaped the world of truly scalable road routing algorithms. 
# From his alumni I distinguish two persons:
#  * [Dominik Schultes](http://algo2.iti.kit.edu/schultes/) who won the [DIMACS challenge on shortest paths]( http://www.diag.uniroma1.it//challenge9/data/tiger/) and made it to the [Scientific American top 50]( https://www.scientificamerican.com/article/sciam-50-the-fastest-way/). Before Dominik’s research scalable shortest paths on large national road networks where heuristics, now they are exact and can be computed at world scale. 
#  * [Dennis Luxen]( http://algo2.iti.kit.edu/english/luxen.php) for creating https://github.com/Project-OSRM/osrm-backend which offers a free, scalable, implementation of [contraction hierarchies]( https://en.wikipedia.org/wiki/Contraction_hierarchies).
#  
# Finally, I mention [Fletcher Foti]( https://fletcherfoti.weebly.com/) who gave us [pandana]( http://udst.github.io/pandana/).
#  
# 

# ## Geocoding
# 
# The world is mapped with the [geographic coordinate system](https://en.wikipedia.org/wiki/Geographic_coordinate_system) but we have difficulties remembering [latitudes]( https://en.wikipedia.org/wiki/Latitude) and [longitudes]( https://en.wikipedia.org/wiki/Longitude).
# 
# We learn and remember the world better from addresses. 
# 

# In[2]:


import geopy
    
def FreeLocator():
    return geopy.Photon(user_agent='myGeocoder')

Amsterdam = FreeLocator().geocode('Amsterdam,NL')


# # Visualization

# In[3]:


import folium
Map = folium.Map(location=(Amsterdam.latitude,Amsterdam.longitude), zoom_start=12)
Map


# In[4]:


def locate_geopy(description):
    location = FreeLocator().geocode(description)
    if location is not None:
        return location.latitude, location.longitude
    return None, None


# In[5]:


import pandas as pd
pd.options.display.float_format = '{:.6f}'.format

data = {'address': [ 'Centraal Station',
                     'Amsterdam Business School',
                     'Artis',
                     'Arena',
                     'Ziggo Dome' ], 
        'color'  : [ 'blue',
                     'black',                   
                     'green',
                     'red',
                     'purple' ]}
# Create DataFrame.
df = pd.DataFrame(data)
df['city']    = 'Amsterdam'
df['country'] = 'NL'
df


# In[6]:


locations = [ locate_geopy(','.join(row[['address','city','country']])) for _, row in df.iterrows() ]
df['lat'] = [ loc[0] for loc in locations ]
df['lon'] = [ loc[1] for loc in locations ]
df


# In[7]:


for _, row in df.iterrows():
    folium.Marker((row.lat,row.lon),icon=folium.Icon(color=row.color), tooltip=row.address).add_to(Map)
Map


# In[8]:


import osmnx as ox
import networkx as nx
from IPython.display import display

ox.config(log_console=True, use_cache=True)


# In[9]:


get_ipython().run_cell_magic('time', '', "G_walk = ox.graph_from_place('Amsterdam, NL', network_type='walk')")


# In[10]:


print( G_walk.number_of_nodes(), G_walk.number_of_edges() )


# In[11]:


df['osmnx'] = ox.distance.nearest_nodes(G_walk,df.lon,df.lat)
df


# In[12]:


get_ipython().run_line_magic('time', "route = nx.shortest_path(G_walk,df.iloc[0].osmnx,df.iloc[1].osmnx,weight='length')")
print(route)


# In[13]:


route_map = ox.plot_route_folium(G_walk, route)
display(route_map)


# In[14]:


get_ipython().run_cell_magic('time', '', "nodes = pd.DataFrame.from_dict(dict(G_walk.nodes(data=True)), orient='index')")


# In[15]:


get_ipython().run_cell_magic('time', '', 'edges = nx.to_pandas_edgelist(G_walk)')


# In[16]:


nodes.street_count.describe()


# In[17]:


edges.length.describe()


# In[18]:


edges.loc[edges.length.idxmax()]


# In[19]:


get_ipython().run_line_magic('time', "longest = nx.shortest_path(G_walk,2632720004,46544942,weight='length')")
print(longest)


# In[20]:


route_map = ox.plot_route_folium(G_walk, longest)
display(route_map)


# # Dijkstra on steroids for road networks

# In[21]:


get_ipython().run_cell_magic('time', '', "import pandana\nnetwork = pandana.Network(nodes['x'], nodes['y'], edges['source'], edges['target'], edges[['length']],twoway=True)")


# In[22]:


network.nodes_df.head()


# In[23]:


network.edges_df.head()


# In[24]:


df['pandana'] = network.get_node_ids(df.lon, df.lat).values
df


# In[25]:


get_ipython().run_line_magic('time', 'path_pandana = network.shortest_path(df.iloc[2].pandana, df.iloc[3].pandana)')


# In[26]:


get_ipython().run_line_magic('time', "path_nx = nx.shortest_path(G_walk,df.iloc[2].osmnx,df.iloc[3].osmnx,weight='length')")


# In[27]:


A = set(path_pandana)
B = set(path_nx)
(A | B) - (A & B)


# In[28]:


origs = [o for o in df.pandana for d in df.pandana]
dests = [d for o in df.pandana for d in df.pandana]
get_ipython().run_line_magic('time', 'distances = network.shortest_path_lengths(origs, dests)')


# In[29]:


import numpy as np 

n = len(df)
pd.options.display.float_format = '{:.2f}'.format
pd.DataFrame(np.array(list(distances)).reshape(n,n),index=df.address,columns=df.address)


# In[30]:


np.random.seed(2021)
n = 500
sample = np.random.choice(np.array(network.nodes_df.index.values.tolist()), n, replace=False)
origs = [o for o in sample for d in sample]
dests = [d for o in sample for d in sample]


# In[31]:


get_ipython().run_line_magic('time', 'distances = network.shortest_path_lengths(origs, dests)')
get_ipython().run_line_magic('time', 'table = pd.DataFrame(np.array(list(distances)).reshape(n,n),index=sample,columns=sample)')


# In[32]:


departure = table.max(axis=1).idxmax()
arrival = table.loc[departure].idxmax()
get_ipython().run_line_magic('time', 'path_pandana = network.shortest_path(departure, arrival)')
get_ipython().run_line_magic('time', "path_nx = nx.shortest_path(G_walk,departure,arrival,weight='length')")
A = set(path_pandana)
B = set(path_nx)
(A | B) - (A & B)


# In[33]:


get_ipython().run_line_magic('time', 'paths = network.shortest_paths(origs,dests)')


# In[34]:


sum(map(len,paths))


# In[35]:


for u,v in zip(paths[1][:-1],paths[1][1:]):
    print(G_walk.get_edge_data(u,v)[0].get('name',''))


# In[36]:


route_map = ox.plot_route_folium(G_walk, paths[1],color='red',map=Map)
display(route_map)

