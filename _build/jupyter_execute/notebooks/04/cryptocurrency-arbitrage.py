#!/usr/bin/env python
# coding: utf-8

# # Crypto Currency Analysis

# ## Bibliographic Notes
# 
# Crytocurrency markets are still a relatively new and relatively few academic papers are available that specifically address arbitrage on those markets. Early studies, such as the following, reported periods of large, recurrent arbitrage opportunities that exist across exchanges, and that can persist for several days or weeks.
# 
# > Makarov, I., & Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. Journal of Financial Economics, 135(2), 293-319.
# 
# Subsequent work reports these prices differentials do exist, but only at a fraction of the values previously reported, and only for fleeting periods of time. 
# 
# > Crépellière, T., & Zeisberger, S. (2020). Arbitrage in the Market for Cryptocurrencies. Available at SSRN 3606053.  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3606053
# 
# The use of network algorithms to identify cross-exchange arbitrage has appeared in the academic literature, and in numerous web sites demonstrating optimization and network applications. Representative examples are cited below.
# 
# > Peduzzi, G., James, J., & Xu, J. (2021, September). JACK THE RIPPLER: Arbitrage on the Decentralized Exchange of the XRP Ledger. In 2021 3rd Conference on Blockchain Research & Applications for Innovative Networks and Services (BRAINS) (pp. 1-2). IEEE. https://arxiv.org/pdf/2106.16158.pdf
# 
# > Bruzgė, R., & Šapkauskienė, A. (2022). Network analysis on Bitcoin arbitrage opportunities. The North American Journal of Economics and Finance, 59, 101562. https://doi.org/10.1016/j.najef.2021.101562
# 
# > Bruzgė, R., & Šapkauskienė, A. (2022). Dataset for Bitcoin arbitrage in different cryptocurrency exchanges. Data in Brief, 40, 107731. 
# 
# The work in this notebook is related to materials found in the following web resources.
# 
# > https://anilpai.medium.com/currency-arbitrage-using-bellman-ford-algorithm-8938dcea56ea
# 
# > [Crypto Trading and Arbitrage Identification Strategies](https://nbviewer.org/github/rcroessmann/sharing_public/blob/master/arbitrage_identification.ipynb)
# 
# A more complete analysis of trading and exploiting arbitrage opportunities in decentralized finance markets is available in the following paper and thesis.
# 
# > Byrne, S. An Exploration of Novel Trading and Arbitrage Methods within Decentralised Finance. https://www.scss.tcd.ie/Donal.OMahony/bfg/202021/StephenByrneDissertation.pdf
# 
# > Levus, R., Berko, A., Chyrun, L., Panasyuk, V., & Hrubel, M. (2021). Intelligent System for Arbitrage Situations Searching in the Cryptocurrency Market. In CEUR Workshop Proceedings (pp. 407-440). http://ceur-ws.org/Vol-2917/paper32.pdf
# 
# In addition to the analysis of arbitrage opportunities, convex optimization may also have an important role in the developing of trading algorithms for crypocurrency exchanges.
# 
# > Angeris, G., Agrawal, A., Evans, A., Chitra, T., & Boyd, S. (2021). Constant function market makers: Multi-asset trades via convex optimization. arXiv preprint arXiv:2107.12484. https://baincapitalcrypto.com/constant-function-market-makers-multi-asset-trades-via-convex-optimization/ and https://arxiv.org/pdf/2107.12484.pdf
# 
# 
# 
# 

# ## Installations and Imports
# 
# This notebook requires multiple libraries. The following cell performs the required installations for Google Colab. To operate your own device you will need to install the `pyomo`,`ccxt`, and `graphviz` python libraries, the graphviz executables, and a linear solver for Pyomo.

# In[1]:


import sys

if "google.colab" in sys.modules:
    get_ipython().system('pip install -q ccxt')
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# In[2]:


import os
from time import time
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from graphviz import Digraph


# ## Accessing Exchange Trading Data
# 
# The open-source library `ccxt` currently supports APIs for 114 exchanges on which cryptocurrencies are traded.

# In[3]:


import ccxt

print(ccxt.exchanges)


# ### Select an exchange and load available symbols
# 
# Each of the exchanges supported by `ccxt` offers multiple **markets**, each market defined by two (possibly more) currencies. `ccxt` labels each market with an id and with a symbol. The market id is unique per exchange and are there for HTTP request-response purposes. The symbol, however, are common across exchanges which make them suitable for arbitrage and other cross-exchange analyses.
# 
# Each symbol is usually an upper case string with names for a pair of traded currencies separated by a slash. The first is called the base currency, and the second the quote currency. Here we refer to the base currency as the source SRC, and the quote currency as the destination DST.
# 
# $$SRC/DST$$
# 
# Associated with each symbol are two potential market transactions.
# 
# * $SRC \rightarrow DST$ converts one unit of the currency desigated by $SRC$ on the exchange to multiple units of the currency designated $DST$ at the 'bid' price.
# * $DST \rightarrow SRC$ converts one unit of $DST$ on to multiple units of $SRC$ at the inverse of the 'ask' price.
# 
# Here we represent the transcations using nodes and edges from graph theory. Each node is designates a currency trading on the exchange. Each edge corresponds to a currency transaction comprised of a source and destination node. Here we assume all currencies are being traded on the same exchange. This notation can be extended to multi-exchange trading by designating nodes as (exchange, currency) pairs.
# 
# 

# In[43]:


import ccxt
import networkx as nx
import pandas as pd

# global variables used in subsequent cells
exchange = ccxt.binanceus()
markets = exchange.load_markets()
symbols = exchange.symbols

# split symbols into a list of SRC -> DST pairs
src_dst_pairs = [symbol.split("/") for symbol in symbols]

# sorted list of all unique currencies
currencies = [currency for src_dst in src_dst_pairs for currency in src_dst]
currencies = sorted(list(set(currencies)))

# sources and destinations dictionaries for currency in currencies
sources = dict()
for currency in currencies:
    sources[currency] = [src for src, dst in src_dst_pairs if dst == currency]

destinations = dict()
for currency in currencies:
    destinations[currency] = [dst for src, dst in src_dst_pairs if src == currency]


# We seek ways of restricting the graph to the most traded or most liquid currencies. Here we identify the "base" currencies as those appearing in the destination list, and keep source currencies that are traded in $N$ or more base currencies.

# In[63]:


import matplotlib.pyplot as plt
import networkx as nx

# list of currencies traded in N or more quote currencies
N = 4
nodes = [
    currency
    for currency in currencies
    if ((len(sources[currency]) >= 1) or (len(destinations[currency]) >= N))
]

dg = nx.DiGraph()
for currency in nodes:
    dg.add_node(currency)
    for src in sources[currency]:
        if src in select:
            dg.add_edge(src, currency)
    for dst in destinations[currency]:
        if dst in select:
            dg.add_edge(currency, dst)
            
edges = dg.edges()
print(f"Number of nodes = {len(nodes):3d}")
print(f"Number of edges = {len(edges):3d}")

fig = plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, iterations=100)
nx.draw(
    dg,
    with_labels=True,
    node_color="lightblue",
    node_size=1000,
    font_size=8,
    arrowsize=10,
)


# ## Order Book

# In[61]:


trade_symbols = ["/".join(edge) for edge in edges]


def fetch_order_book(symbol, limit=1, exchange=exchange):
    """return order book data for a specified symbol"""
    start_time = timer()
    result = exchange.fetch_order_book(symbol, limit)
    result["src"], result["dst"] = symbol.split("/")
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
print(
    f"fetched order book for {len(fetched_order_book)} symbols in {run_time:0.3f} seconds"
)

# convert to pandas dataframe
order_book = pd.DataFrame(fetched_order_book).T
order_book.drop(columns=["datetime", "symbol", "bids", "asks"], inplace=True)
order_book["timestamp"] = pd.to_datetime(order_book["timestamp"], unit="ms")
order_book


# ## Create directed graph of the order book

# In[64]:


# dictionary of edges indexed by (src, dst) tuples
#     type: 'bid' or 'ask'
#     conv: 1 unit of src currency produces conv units of dst currency
#     log10_conv: log10 of conv
bids = {
    (order_book.at[symbol, "src"], order_book.at[symbol, "dst"]): {
        "type": "bid",
        "conv": order_book.at[symbol, "bid_price"],
        "log10_conv": np.log10(order_book.at[symbol, "bid_price"]),
        #'volume': order_book.at[symbol, 'bid_volume'],
    }
    for symbol in order_book.index
    if not np.isnan(order_book.at[symbol, "bid_volume"])
}

asks = {
    (order_book.at[symbol, "dst"], order_book.at[symbol, "src"]): {
        "type": "ask",
        "conv": 1.0 / order_book.at[symbol, "ask_price"],
        "log10_conv": np.log10(1.0 / order_book.at[symbol, "ask_price"]),
        #'volume': 1.0/order_book.at[symbol, 'ask_volume'],
    }
    for symbol in order_book.index
    if not np.isnan(order_book.at[symbol, "ask_volume"])
}


# In[67]:


edges = dict()
edges.update(bids)
edges.update(asks)
print(len(edges))


# ## Graphviz
# 
# https://graphviz.readthedocs.io/en/stable/
# 

# In[10]:


# plot a directed graph from the edges and nodes

timestamp_min = order_book["timestamp"].min()
timestamp_max = order_book["timestamp"].max()

label = (
    f"{exchange} \nBest Bid/Ask Order Book\n{timestamp_min} - {timestamp_max}"
    + "\nCurrencies trading on {N} or more base currencies"
    + "\nEdges labeled with log base 10 of conversion"
    + "\n "
)
dot = Digraph(
    graph_attr={"label": label, "fontsize": "15", "labelloc": "t"},
    node_attr={"fontsize": "10"},
    edge_attr={"fontsize": "10"},
)

for node, prop in nodes.items():
    if prop["type"] == "dst":
        dot.node(node, style="filled", fillcolor="gold")
    else:
        dot.node(node, style="filled", fillcolor="lightblue")

for edge, prop in edges.items():
    src, dst = edge
    log10_conv = prop["log10_conv"]
    label = f"{log10_conv:0.5f}"
    if prop["type"] == "bid":
        dot.edge(src, dst, label, color="black", fontcolor="black")
    else:
        dot.edge(src, dst, label, color="red", fontcolor="red")

display(dot)
dot.format = "png"
dot.view("exchange-dag")


# In[11]:


# split the graph on USD ...

nodes["USD-SRC"] = {"type": "src"}
nodes["USD-DST"] = {"type": "dst"}
nodes.pop("USD", None)

for edge in list(edges.keys()):
    src, dst = edge
    if src == "USD":
        value = edges[edge]
        edges[("USD-SRC", dst)] = value

    elif dst == "USD":
        value = edges[edge]
        edges[(src, "USD-DST")] = value

for edge in list(edges.keys()):
    src, dst = edge
    if (src == "USD") or (dst == "USD"):
        edges.pop(edge)


# In[12]:


label = (
    f"{exchange} \nBest Bid/Ask Order Book\n{timestamp_min} - {timestamp_max}"
    + f"\nCurrencies trading on {N} or more base currencies"
    + "\nEdges labeled with log base 10 of conversion"
    + "\n "
)
dot = Digraph(
    graph_attr={"label": label, "fontsize": "15", "labelloc": "t"},
    node_attr={"fontsize": "10"},
    edge_attr={"fontsize": "10"},
)

for node, prop in nodes.items():
    if prop["type"] == "dst":
        dot.node(node, style="filled", fillcolor="gold")
    else:
        dot.node(node, style="filled", fillcolor="lightblue")

for edge, prop in edges.items():
    src, dst = edge
    log10_conv = prop["log10_conv"]
    label = f"{log10_conv:0.5f}"
    if prop["type"] == "bid":
        dot.edge(src, dst, label, color="black", fontcolor="black")
    else:
        dot.edge(src, dst, label, color="red", fontcolor="red")

display(dot)
dot.format = "png"
dot.view("exchange-dag")


# In[13]:


import pyomo.environ as pyo

T = 10

m = pyo.ConcreteModel(f"{exchange} arbitrage")

# length of the trading chain
m.T0 = pyo.RangeSet(0, T)
m.T1 = pyo.RangeSet(1, T)

# currency nodes and trading edges
m.NODES = pyo.Set(initialize=nodes.keys())
m.EDGES = pyo.Set(initialize=edges.keys())

# "gain" on each trading edge
@m.Param(m.EDGES)
def a(m, src, dst):
    return 10 ** edges[(src, dst)]["log10_conv"]


# currency on hand at each node
m.w = pyo.Var(m.NODES, m.T0, domain=pyo.NonNegativeReals)

# amount traded on each edge
m.x = pyo.Var(m.EDGES, m.T1, domain=pyo.NonNegativeReals)


@m.Objective(sense=pyo.maximize)
def wealth(m):
    return m.w["USD-DST", T]


# initial assignment of 100 units on a selected currency
@m.Constraint(m.NODES)
def initial(m, node):
    if node == "USD-SRC":
        return m.w[node, 0] == 1.0
    return m.w[node, 0] == 0.0


@m.Constraint(m.NODES, m.T1)
def no_shorting(m, node, t):
    return m.w[node, t - 1] >= sum(
        m.x[node, dst, t] for src, dst in m.EDGES if src == node
    )


@m.Constraint(m.NODES, m.T1)
def balances(m, node, t):
    return m.w[node, t] == m.w[node, t - 1] - sum(
        m.x[node, dst, t] for src, dst in m.EDGES if src == node
    ) + sum(m.a[src, node] * m.x[src, node, t] for src, dst in m.EDGES if dst == node)


solver = pyo.SolverFactory("cbc")
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


# In[14]:


import math

n = len(nodes)


def visualize(bf):
    adot = Digraph()
    for node in bf.keys():
        w = 10 ** (-bf[node]["w"])
        if math.isinf(w):
            adot.node(node)
        else:
            adot.node(node, label=f"{node}\n{w:0.8f}")
        if bf[node]["prec"] is not None:
            adot.edge(bf[node]["prec"], node)
    display(adot)


# dictionary of all nodes.
#   prec: preceding node
#   w: weight

bf = {node: {"prec": None, "w": float("inf")} for node in nodes}
bf["USD-SRC"]["w"] = 0

for _ in range(n - 1):
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

# In[15]:


get_ipython().run_cell_magic('script', 'echo skipping', "\nfrom timeit import default_timer as timer\n\n# asynchronous implementation\nimport asyncio\nimport nest_asyncio\n\nmy_symbols = ['/'.join(edge) for edge in edges]\n\nasync def afetch_order_book(symbol, limit=1, exchange=exchange):\n    start_time = timer()\n    result = exchange.fetch_order_book(symbol, limit)\n    run_time = timer() - start_time\n    return result\n\nasync def get_data():\n    coroutines = [afetch_order_book(symbol) for symbol in my_symbols]\n    await asyncio.gather(*coroutines)\n\nstart = timer()\nnest_asyncio.apply()\nasyncio.run(get_data())\nrun_time = timer() - start\n\nprint(run_time)")


# In[15]:





# In[7]:


# trim currencies to those that appears as DST, or are N or more SRC

# all currencies trading in N or more base currencies
N = 4

src_nodes = list(src_count[src_count > N].index)
dst_nodes = list(dst_count[dst_count > 1].index)
src_dst_nodes = list(set(src_nodes + dst_nodes))

# plot a directed graph from the edges and nodes
label = f"{exchange}\ntrading symbols with {N} or more base currencies\n "
dg = Digraph(
    f"{exchange}",
    graph_attr={"label": label, "fontsize": "15", "labelloc": "t"},
    node_attr={"fontsize": "12"},
    edge_attr={"fontsize": "10"},
)

for node in src_nodes:
    label = f"{node}"
    dg.node(node, label, style="filled", fillcolor="lightblue")

for node in dst_nodes:
    label = f"{node}"
    dg.node(node, label, style="filled", fillcolor="gold")

trade_edges = [
    [src, dst]
    for src, dst in src_dst_pairs
    if (src in src_dst_nodes) and (dst in src_dst_nodes)
]
for src, dst in trade_edges:
    symbol = "/".join([src, dst])
    label = f"{symbol}"
    dg.edge(src, dst, label)

display(dg)
dg.format = "png"
dg.view("exchange-symbol-map")

