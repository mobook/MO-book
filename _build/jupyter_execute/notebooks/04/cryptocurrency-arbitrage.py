#!/usr/bin/env python
# coding: utf-8

# # Crypto Currency Analysis
# 
# https://anilpai.medium.com/currency-arbitrage-using-bellman-ford-algorithm-8938dcea56ea

# ## Installations and Imports

# In[ ]:


import sys
if "google.colab" in sys.modules:
    get_ipython().system('pip install -q ccxt')
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# In[ ]:


import ccxt
import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo

from time import time
from timeit import default_timer as timer
from graphviz import Digraph


# ## Select an exchange and load available symbols

# In[ ]:


# set the exchange ... careful of generators here
exchange = ccxt.binanceus()
markets = exchange.load_markets()
symbols = exchange.symbols
exch = str(exchange)


# ## Interpreting Trading Symbols as Directed Graphs
# 
# Each market symbol on an exchange describes currency transactions that we'll characterize with a source/destination pair. The exchange symbol
# 
# $$SRC/DST$$
# 
# describes two potential transactions:
# 
# * $SRC \rightarrow DST$ converts one unit $SRC$ on that exchange to multiple units of $DST$ equal to 'bid' price on that exchange, and
# * $DST \rightarrow SRC$ converts one unit of $DST$ on to multiple units of $SRC$ at the inverse of the 'ask' price.
# 
# The transactions are specific to the associated exchange.
# 
# Here we represent the transcations using nodes and edges from graph theory. Each 
# 
# From the sources and destinations we create a set of nodes that will be indexed by (EXCH, CURR).
# 
# 

# In[ ]:


# split symbols into SRC -> DST arcs and create list of currencies

# lists of all currencies and trades that appear in symbols
src_dst_pairs = [symbol.split('/') for symbol in symbols]
currencies = sorted(list(set([currency for src_dst in src_dst_pairs for currency in src_dst])))

# count times each currency as a src and a dst 
src_count = pd.Series({currency: sum(1 for src, dst in src_dst_pairs if currency==src)  for currency in currencies})
dst_count = pd.Series({currency: sum(1 for src, dst in src_dst_pairs if currency==dst)  for currency in currencies})

# which currencies appear as SRC
print("\nCurrencies appearing one more times as SRC")
print(src_count[src_count >= 1])

# which currencies appear as DST
print("\nCurrencies appearing one more times as DST")
print(dst_count[dst_count >= 1])


# We seek ways of restricting the graph to the most traded or most liquid currencies. Here we identify the "base" currencies as those appearing in the destination list, and keep source currencies that are traded in $N$ or more base currencies.

# In[ ]:


# trim currencies to those that appears as DST, or are N or more SRC

# all currencies trading in N or more base currencies
N = 1

src_nodes = list(src_count[src_count>N].index)
dst_nodes = list(dst_count[dst_count>1].index)
src_dst_nodes = list(set(src_nodes + dst_nodes))

# plot a directed graph from the edges and nodes
label = f"{exch}\ntrading symbols with {N} or more base currencies\n "
dg = Digraph(f"{exch}", 
             graph_attr={'label': label, 'fontsize': '15', 'labelloc': 't'},
             node_attr={'fontsize': '12'},
             edge_attr={'fontsize': '10'})

for node in src_nodes:
    label = f"{node}"
    dg.node(node, label, style='filled', fillcolor='lightblue')

for node in dst_nodes:
    label = f"{node}"
    dg.node(node, label, style='filled', fillcolor='gold')

trade_edges = [[src, dst] for src, dst in src_dst_pairs if (src in src_dst_nodes) and (dst in src_dst_nodes)]
for src, dst in trade_edges:
    symbol = '/'.join([src, dst])
    label = f"{symbol}"
    dg.edge(src, dst, label)

display(dg)
dg.format = "png"
dg.view("exchange-symbol-map")


# ## Order Book

# In[ ]:


trade_symbols = ['/'.join(edge) for edge in trade_edges]

def fetch_order_book(symbol, limit=1, exchange=exchange):
    """return order book data for a specified symbol"""
    start_time = timer()
    result = exchange.fetch_order_book(symbol, limit)
    result["src"], result["dst"] = symbol.split('/')
    result["run_time"] = timer() - start_time
    result["timestamp"] = exchange.milliseconds()
    if result["bids"]:
        result["bid_price"] = result["bids"][0][0]
        result["bid_volume"] = result["bids"][0][1]
    if result["asks"]:
        result["ask_price"] = result["asks"][0][0]
        result["ask_volume"] = result["asks"][0][1]
    return result

# fetch order book data and store in a dictionay
start = timer()
fetched_order_book = {symbol: fetch_order_book(symbol) for symbol in trade_symbols}
run_time = timer() - start
print(f"fetched order book for {len(fetched_order_book)} symbols in {run_time:0.3f} seconds")

# convert to pandas dataframe
order_book = pd.DataFrame(fetched_order_book).T
order_book.drop(columns=["datetime", "symbol", "bids", "asks"], inplace=True)
order_book['timestamp'] = pd.to_datetime(order_book['timestamp'], unit='ms')
order_book


# ## Create directed graph of the order book

# In[ ]:


# dictionary of currencies
nodes = dict()
nodes.update({node: {"type": "src"} for node in src_nodes})
nodes.update({node: {"type": "dst"} for node in dst_nodes})


# In[ ]:



# dictionary of edges indexed by (src, dst) tuples
#     type: 'bid' or 'ask'
#     conv: 1 unit of src currency produces conv units of dst currency
#     log10_conv: log10 of conv
bids = {(order_book.at[symbol, 'src'], order_book.at[symbol, 'dst']): 
        {'type': 'bid', 
         'conv': order_book.at[symbol, 'bid_price'],
         'log10_conv': np.log10(order_book.at[symbol, 'bid_price']),
         #'volume': order_book.at[symbol, 'bid_volume'],
        }
        for symbol in order_book.index if not np.isnan(order_book.at[symbol, "bid_volume"])
}

asks = {(order_book.at[symbol, 'dst'], order_book.at[symbol, 'src']): 
        {'type': 'ask', 
         'conv': 1.0/order_book.at[symbol, 'ask_price'],
         'log10_conv': np.log10(1.0/order_book.at[symbol, 'ask_price']),
         #'volume': 1.0/order_book.at[symbol, 'ask_volume'],
        }
        for symbol in order_book.index if not np.isnan(order_book.at[symbol, "ask_volume"])
}

edges = dict()
edges.update(bids)
edges.update(asks)


# ## Graphviz
# 
# https://graphviz.readthedocs.io/en/stable/
# 

# In[ ]:


# plot a directed graph from the edges and nodes

timestamp_min = order_book['timestamp'].min()
timestamp_max = order_book['timestamp'].max()

label = f"{exch} \nBest Bid/Ask Order Book\n{timestamp_min} - {timestamp_max}" +         "\nCurrencies trading on {N} or more base currencies" +         "\nEdges labeled with log base 10 of conversion" +         "\n "
dot = Digraph(
        graph_attr={'label': label, 'fontsize': '15', 'labelloc': 't'},
        node_attr={'fontsize': '10'},
        edge_attr={'fontsize': '10'},
    )

for node, prop in nodes.items():
    if prop["type"] == "dst":
        dot.node(node, style='filled', fillcolor='gold')
    else:
        dot.node(node, style='filled', fillcolor='lightblue')

for edge, prop in edges.items():
    src, dst = edge
    log10_conv = prop['log10_conv']
    label=f"{log10_conv:0.5f}"
    if prop["type"] == "bid":
        dot.edge(src, dst, label, color='black', fontcolor='black')
    else:
        dot.edge(src, dst, label, color='red', fontcolor='red')   

display(dot)
dot.format = "png"
dot.view("exchange-dag")


# In[ ]:


# split the graph on USD ... 

nodes['USD-SRC'] = {'type': 'src'}
nodes['USD-DST'] = {'type': 'dst'}
nodes.pop('USD', None)

for edge in list(edges.keys()):
    src, dst = edge
    if src == "USD":
        value = edges[edge]
        edges[('USD-SRC', dst)] = value

    elif dst == "USD":
        value = edges[edge]
        edges[(src, 'USD-DST')] = value

for edge in list(edges.keys()):
    src, dst = edge
    if (src == 'USD') or (dst == 'USD'):
        edges.pop(edge)


# In[ ]:




label = f"{exch} \nBest Bid/Ask Order Book\n{timestamp_min} - {timestamp_max}" +         f"\nCurrencies trading on {N} or more base currencies" +         "\nEdges labeled with log base 10 of conversion" +         "\n "
dot = Digraph(
        graph_attr={'label': label, 'fontsize': '15', 'labelloc': 't'},
        node_attr={'fontsize': '10'},
        edge_attr={'fontsize': '10'},
    )

for node, prop in nodes.items():
    if prop["type"] == "dst":
        dot.node(node, style='filled', fillcolor='gold')
    else:
        dot.node(node, style='filled', fillcolor='lightblue')

for edge, prop in edges.items():
    src, dst = edge
    log10_conv = prop['log10_conv']
    label=f"{log10_conv:0.5f}"
    if prop["type"] == "bid":
        dot.edge(src, dst, label, color='black', fontcolor='black')
    else:
        dot.edge(src, dst, label, color='red', fontcolor='red')   

display(dot)
dot.format = "png"
dot.view("exchange-dag")


# In[ ]:


import pyomo.environ as pyo

T = 10

m  = pyo.ConcreteModel(f"{exch} arbitrage")

# length of the trading chain
m.T0 = pyo.RangeSet(0, T)
m.T1 = pyo.RangeSet(1, T)

# currency nodes and trading edges
m.NODES = pyo.Set(initialize=nodes.keys())
m.EDGES = pyo.Set(initialize=edges.keys())

# "gain" on each trading edge
@m.Param(m.EDGES)
def a(m, src, dst):
    return 10**edges[(src, dst)]["log10_conv"]

# currency on hand at each node
m.w = pyo.Var(m.NODES, m.T0, domain=pyo.NonNegativeReals)

# amount traded on each edge
m.x = pyo.Var(m.EDGES, m.T1, domain=pyo.NonNegativeReals)

@m.Objective(sense=pyo.maximize)
def wealth(m):
    return m.w['USD-DST', T]

# initial assignment of 100 units on a selected currency
@m.Constraint(m.NODES)
def initial(m, node):
    if node == 'USD-SRC':
        return m.w[node, 0] == 1.0
    return m.w[node, 0] == 0.0

@m.Constraint(m.NODES, m.T1)
def no_shorting(m, node, t):
    return m.w[node, t-1] >= sum(m.x[node, dst, t] for src, dst in m.EDGES if src==node)

@m.Constraint(m.NODES, m.T1)
def balances(m, node, t):
    return m.w[node, t] == m.w[node, t-1] - sum(m.x[node, dst, t] for src, dst in m.EDGES if src==node)                                           + sum(m.a[src, node]*m.x[src, node, t] for src, dst in m.EDGES if dst==node)

solver = pyo.SolverFactory('cbc')
solver.solve(m)

for node in m.NODES:
    print(f"{node:8s}", end="")
    for t in m.T0:
        print(f" {m.w[node, t]():12.8f}", end="")
    print()

print()

for t in m.T1:
    print(f"t = {t}")
    for src, dst in m.EDGES:
        if m.x[src, dst, t]() > 0:
            print(f"{src:8s} -> {dst:8s}: {m.x[src, dst, t]():0.8f}")
    print()


# In[ ]:


import math

n = len(nodes)

def visualize(bf):
    adot = Digraph()
    for node in bf.keys():
        w = 10**(-bf[node]["w"])
        if math.isinf(w):
            adot.node(node)
        else:
            adot.node(node, label=f"{node}\n{w:0.8f}")
        if bf[node]['prec'] is not None:
            adot.edge(bf[node]['prec'], node)
    display(adot)

# dictionary of all nodes. 
#   prec: preceding node
#   w: weight

bf = {node: {'prec': None, 'w': float('inf')} for node in nodes}
bf["USD-SRC"]["w"] = 0

for _ in range(n-1):
    for edge in edges.keys():
        src, dst = edge
        w = -edges[edge]["log10_conv"]
        if bf[dst]["w"] > (bf[src]["w"] + w):
            bf[dst]["w"] = bf[src]["w"] + w
            bf[dst]["prec"] = src

visualize(bf)


# ## Appendix: Asynchronous downloads of order book data
# 
# Considerably more development, testing, and validation would be needed to adapt this code to an automated tradign bot to exploit arbitrage opportunities in the crypto-currency markets. One of the time consuming steps is accessing order book data. The following cell is an experiment using the Python `asyncio` library to determine if asynchronous calls to the `ccxt` would provide faster downloads. 
# 
# Preliminary testing shows little or no advantage to `asyncio` when working with a single exchange. That may change when the above code is adapted to multi-exchange arbitrage, and therefore this code is retained below for future testing.

# In[ ]:


get_ipython().run_cell_magic('script', 'echo skipping', "\nfrom timeit import default_timer as timer\n\n# asynchronous implementation\nimport asyncio\nimport nest_asyncio\n\nmy_symbols = ['/'.join(edge) for edge in edges]\n\nasync def afetch_order_book(symbol, limit=1, exchange=exchange):\n    start_time = timer()\n    result = exchange.fetch_order_book(symbol, limit)\n    run_time = timer() - start_time\n    return result\n\nasync def get_data():\n    coroutines = [afetch_order_book(symbol) for symbol in my_symbols]\n    await asyncio.gather(*coroutines)\n\nstart = timer()\nnest_asyncio.apply()\nasyncio.run(get_data())\nrun_time = timer() - start\n\nprint(run_time)")


# In[ ]:




