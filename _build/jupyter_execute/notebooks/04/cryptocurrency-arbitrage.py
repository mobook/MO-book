#!/usr/bin/env python
# coding: utf-8

# ```{index} single: application; cryptocurrency arbitrage
# ```
# ```{index} single: solver; cbc
# ```
# ```{index} web scraping
# ```
# ```{index} pandas dataframe
# ```
# ```{index} networkx
# ```
# ```{index} network optimization
# ```
# 
# # Cryptocurrency arbitrage search
# 
# Crpytocurrency exchanges are web services for the purchase and sale of cryptocurrencies. These exchanges provide liquidity for owners and establish the relative value of among currencies. As of this writing (early 2023), [cryptocurrencies have an estimated collective market capitalization over one trillion USD](https://www.statista.com/statistics/730876/cryptocurrency-maket-value/). Cryptocurrency markets are constantly changing with the introduction of new exchanges, new currencies, the occasional collapse currencies, and highly volatile prices.
# 
# This notebook explores the market efficiency of cryptocurrency exchanges by testing for arbitrage opportunities. Arbitrage happens when a trader realizes a risk-free profit through a sequence of trades. The efficient market hypothesis suggests arbitrage opportunities are quickly identified and exploited by traders. As a result of trading activity, prices maintain a dynamic equilibrium where arbitrage opportunities are short and fleeting. The question explored here is whether real-time and fast execution make it is feasible for a trader to profit from these fleeting arbitrage opportunities.

# ## Installations and Imports
# 
# 
# ### Pyomo and Solvers
# 
# First we import Pyomo and necessary solvers.

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ### CCXT
# 
# In addition to Pyomo and other standard Python libraries, this notebook uses the [open-source library `ccxt`](https://github.com/ccxt/ccxt). `ccxt` supports the real-time APIs of the largest and most common exchanges on which cryptocurrencies are traded. The library can be installed with
# 
#     !pip install ccxt
# 
# Here we import all needed libraries and `ccxt`.

# In[2]:


import os
import sys
from time import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo


# ### Networkx
# 
# This notebook uses the [networkx](https://networkx.org/) library to display exchange and market data. Networkx has been updated recently to version 3.0, but that update has not yet propogated through common Python distributions. If the code that follows in this notebook may generate errors on displaying networkx diagrams, it may be necessary to update the networkx library. This can be done with the following command executed in a new cell.
# 
#     !pip install networkx --upgrade

# ## Cryptocurrency exchanges
# 
# Cryptocurrency exchanges are digital marketplaces for buying and trading cryptocurrencies. Joining an exchange enables a member to maintain multiple currencies in a digital wallet, to buy and sell currencies, and to use cryptocurrencies for financial transactions.  Here we import the library and list current exchanges supported by `ccxt`.

# In[3]:


import ccxt

print("Available exchanges:\n")
for i, exchange in enumerate(ccxt.exchanges):
    print(f"{i+1:3d}) {exchange.ljust(20)}", end="" if (i+1) % 4 else "\n")


# ## Representing an exchange as a directed graph
# 
# Cryptocurrency exchanges facilitate trading between different cryptocurrencies through markets, which are labeled with symbols common across all exchanges. These symbols consist of an upper case string with abbreviations for the traded currency pair, separated by a slash (/). The first abbreviation is the base currency and the second is the quote currency. Prices for the base currency are denominated in units of the quote currency. Each market symbol can refer to a *bid* (an offer to buy a specified quantity of the base currency for no more than than a specified price) or to an *ask* (an offer to sell a specified quantity of the base currency for no less than a specified price).
# 
# Market symbols can be used to construct a directed graph representing an exchange, where nodes correspond to currencies and edges correspond to market symbols. The source node of an edge represents the quote currency, and the destination node represents the base currency.
# 
# The in-degree of a node indicates the number of incoming edges, while the out-degree represents the number of outgoing edges. Nodes with out-degrees greater than zero are currencies used to quote the price of other currencies and are highlighted. Nodes with in-degree below a specified threshold, called the `minimum_in_degree`, have fewer markets and are not displayed or analyzed further.
# 

# In[4]:


# global variables used in subsequent cells

# create an exchange object
exchange = ccxt.binanceus()

def get_dg(exchange, minimum_in_degree=1):
    """
    Return a directed graph constructed from the market symbols on a specified exchange.
    """
    markets = exchange.load_markets()
    symbols = markets.keys()

    # create an edge for all market symbols
    dg = nx.DiGraph()
    for base, quote in [symbol.split("/") for symbol in symbols]:
        dg.add_edge(quote, base, color="k", width=1)
    
    # remove base currencies with in_degree less than minimum_in_degree
    remove_nodes = [node for node in dg.nodes if dg.out_degree(node) == 0 and dg.in_degree(node) < minimum_in_degree]
    dg.remove_nodes_from(remove_nodes)
    
    # color quote currencies in gold
    for node in dg.nodes():
        dg.nodes[node]["color"] = "gold" if dg.out_degree(node) > 0 else "lightblue"

    return dg

def draw_dg(dg, rad=0.0):
    """
    Draw directed graph of markets symbols.
    """
    n_nodes = len(dg.nodes)
    size = int(2.5 * np.sqrt(n_nodes))
    fig = plt.figure(figsize=(size, size))
    pos = nx.circular_layout(dg)
    nx.draw(
        dg, 
        pos,
        with_labels=True,
        node_color=[dg.nodes[node]["color"] for node in dg.nodes()],
        edge_color=[dg.edges[u, v]["color"] for u, v in dg.edges],
        width=[dg.edges[u, v]["width"] for u, v in dg.edges],
        node_size=1000,
        font_size=8,
        arrowsize=15,
        connectionstyle=f"arc3, rad={rad}",
    )
    nx.draw_networkx_edge_labels(dg, pos, edge_labels={(src, dst): f"{src}/{dst}" for src, dst in dg.edges()})

    return plt.gca()

minimum_in_degree = 5
dg = get_dg(exchange, minimum_in_degree)
ax = draw_dg(dg, 0.01)
ax.set_title(exchange.name + "\n" + f"Minimum in Degree (Base Currencies) = {minimum_in_degree}")

print(f"Number of nodes = {len(dg.nodes()):3d}")
print(f"Number of edges = {len(dg.edges()):3d}")


# ## Exchange order book

# The **order book** is the real-time inventory of trading orders on an given exchange. 
# 
# A **bid** is an order to buy up to a specified amount of the base currency. The price is not to exceed the 'bid price' specified in the quote currency. The exchange attempts to match the bid to a sell order at a price less than or equal to the bid price. If a transaction occurs the  buyer receives an amount of base currency less than or equal to the bid volume at a price less than or equal to the bid price.
# 
# An **ask** is an offer to sell up to a specified amount of the base currency for a price no less than a specified value  given in the quote currency. If a transaction occurs, then seller will sell no more than a specified about of the base currency at a price no less than the specified value. 
# 
# The exchange order book maintains a list of all active orders for symbols traded on the exchange. Incoming bids above the lowest ask or incoming asks below the highest bid will be immediately matched and transactions executed following rules of the exchange. 
# 
# The follow cell reads a previously saved order book.  Cells at the end of this notebook demonstrate how to retrieve an order book from an exchange and save it as a Pandas DataFrame.

# In[5]:


import glob

# find all previously saved order books
fnames = sorted(glob.glob(f"*orderbook*".replace(" ","_")))
fname = fnames[-1]

# read the oldest
print(f"\nReading: {fname}\n")
order_book = pd.read_csv(fname, index_col=0)
display(order_book)


# ## Representing the order book also as a directed graph

# Like the exchange itself, the order book for the exchange can be represented as a directed graph where nodes correspond to individual currencies, and the edges correspond to trades available in the current order book. Here we outline the construction of a directed encoding the information available from the order book.
# 
# A bid appearing in the order book for market symbol $b/q$ is an order from a prospective counter party to purchase an amount of the base currency $b$ at a bid price given in a quote currency. For a currency trader, a bid in the order book is an opportunity to convert the base currency $b$ into the quote currency $q$. 
# 
# ### Representing bids
# 
# Let $V_b$ and $V_q$ denote the amount of each currency held by the currency trader prior to a transaction. Let $x_{b\rightarrow q}(t)$ denote the amount of currency $b$ converted to $q$ during the transaction. For each unit of $b$ the trader sells to the bidder, the trader receives an amount of currency $q$ equal to the bid price. Therefore, the change in $V_b$ and $V_q$ are given by
# 
# $$\begin{align*}
# \Delta V_b & = - x_{b\rightarrow q} \\
# \Delta V_q & = a_{b\rightarrow q} x_{b\rightarrow q}
# \end{align*}
# $$
# 
# where $a_{b\rightarrow q}$ is a coefficient equal to the bid price for $b$ expressed in units of the quote currency $q$. The capacity $c_{b\rightarrow q}$ of an edge $b\rightarrow q$ is specified by a relationship
# 
# $$x_{b\rightarrow q} \leq c_{b\rightarrow q}$$
# 
# Thus each bid on the order book for symbol $b/q$ results in edge labeled by
# 
# $$
# \begin{align*}
# a_{b\rightarrow q} & = \text{bid price} \\
# c_{b\rightarrow q} & = \text{bid volume}
# \end{align*}
# $$
# 
# ### Representing asks
# 
# An 'ask' for symbol $b/q$, is an order on the exchange to sell the base currency at price not less than the 'ask' price given in terms of the quote currency. The ask volume is the amount of base currency to be sold. For a currency trader, a sell order is an opportunity to convert the quote current into the base currency such that
# 
# $$
# \begin{align*}
# a_{q\rightarrow b} & = \frac{1}{\text{ask price}} \\
# c_{q\rightarrow b} & = \text{ask volume} \times \text{ask volume}
# \end{align*}
# $$
# 
# The following cell creates a directed graph using data from an exchange order book.

# In[6]:


import numpy as np
import networkx as nx

def order_book_to_dg(order_book):
    """
    Convert an order book dataframe into a directed graph using the NetworkX library.

    Parameters:
    -----------
    order_book : pandas.DataFrame
        A dataframe containing the order book information.

    Returns:
    --------
    dg_order_book : networkx.DiGraph
        A directed graph representing the order book.
    """
    
    # create a dictionary of edges index by (src, dst)
    dg_order_book = nx.DiGraph()
    
    # loop over each order in the order book dataframe
    for order in order_book.index:
        # if the order is a 'bid', i.e., an order to purchase the base currency
        if not np.isnan(order_book.at[order, "bid_volume"]):
            src = order_book.at[order, "base"]
            dst = order_book.at[order, "quote"]
            # add an edge to the graph with the relevant attributes
            dg_order_book.add_edge(src, dst,
                kind = "bid",
                a = order_book.at[order, "bid_price"],
                capacity = order_book.at[order, "bid_volume"],
                weight = - np.log(order_book.at[order, "bid_price"]),
                color = "g",
                width = 0.5,
            )

        # if the order is an 'ask', i.e., an order to sell the base currency
        if not np.isnan(order_book.at[order, "ask_volume"]):
            src = order_book.at[order, "quote"]
            dst = order_book.at[order, "base"]
            # add an edge to the graph with the relevant attributes
            dg_order_book.add_edge(src, dst,
                kind = "ask",
                a = 1.0 / order_book.at[order, "ask_price"],
                capacity = order_book.at[order, "ask_volume"] * order_book.at[order, "ask_price"],
                weight = - np.log(1.0 / order_book.at[order, "ask_price"]),
                color = "r",
                width = 0.5,
            )

    # loop over each node in the graph and set the color attribute to "lightblue"
    for node in dg_order_book.nodes():
        dg_order_book.nodes[node]["color"] = "lightblue"

    return dg_order_book

dg_order_book = order_book_to_dg(order_book)


# In[7]:


# display contents of the directed graph
print(f"src   --> dst    kind            a                c")
print(f"------------------------------------------------------")
for src, dst in dg_order_book.edges():
    print(f"{src:5s} --> {dst:5s}   {dg_order_book.edges[(src, dst)]['kind']}" +
          f"{dg_order_book.edges[(src, dst)]['a']: 16f} {dg_order_book.edges[(src, dst)]['capacity']: 16f}    ")


# In[8]:


ax = draw_dg(dg_order_book, 0.05)
ax.set_title("""
Direct Graph for an Order Book

Green: Trader accepts a bid order
Red: Trader accepts a sell order
""");


# ## Trading and Arbitrage
# 
# ### Paths and cycles
# 
# A path in a directed graph constructed from an exchange order book corresponds to a sequence of trades that convert a source currency to the target currency. Given a sequence of nodes $i_0, i_1, \ldots, i_n$, a path of length $n$ is given by
# 
# $$i_0 \rightarrow i_1 \rightarrow i_2 \rightarrow \cdots \rightarrow i_{n-1} \rightarrow i_n$$
# 
# A path is **simple** if there all nodes are distinct. A path is **closed** if it starts and ends at the same node, that is if $i_0 = i_n$. A **simple cycle** is closed, simple path.
# 
# The path has non-zero capacity if each edge on the path has non-zero capacity. For a sufficiently small holding of the source currency, $V_{i_0}$, the result of a sequence of trades will not be affected by the capacity limits. In that case the amount of target currency $i_n$ held after completing $n$ trades will be
# 
# $$V_{i_0 \rightarrow i_i \rightarrow \cdots \rightarrow i_n} = \left(\prod_{k=0}^{n-1} a_{i_k\rightarrow i_{k+1}}\right) V_{i_0}$$
# 
# This amount is path dependent. There may be multiple paths from a source $i_0$ to a target $i_n$. We seek paths from a source $i_0$ to a target $i_n$ that maximize the resulting value. 
# 
# ### Arbitrage
# 
# An arbitrage is a series of trades resulting in risk-free profit. If the order book contains a cycle that starts and ends at $i_0 = i_n$ where
# 
# $$\prod_{k=0}^{n-1} a_{i_k\rightarrow i_{k+1}} > 1$$
# 
# then an arbitrage exists for any currency on the cycle. The size of the arbitrage opportunity will be limited by the capacity available on the edges making up the cycle.
# 
# ### A cycle with negative log returns indicates arbitrage
# 
# To facilitate use of the linear algorithms from the NetworkX library, we assign to each edge a weight equal to the negative logarithm of the conversion coefficients. That is, we let
# 
# $$w_{i\rightarrow j} = - \log a_{i\rightarrow j}$$
# 
# If $w_{i\rightarrow j}$ is interpreted as length of the edge from $i$ to $j$, then a cycle with negative length where  
# 
# $$\sum_{k=0}^{n-1} w_{i_k\rightarrow i_{k+1}} < 0$$
# 
# corresponds to an arbitrage opportunity. For a given order book, we would like to find cycles with negative length where the start and end point is a stable, investment quality currency. 
# 
# ### Testing for a negative cycle
# 
# The Bellman-Ford test will show the existence of at least one simple cycle demonstrating arbitrage. There may be other cycles with higher or lower return, and that allow for higher or lower trading volumes. The NetworkX function [ `negative_edge_cycle` returns true if a negative edge cycle exists somewhere in a directed graph.](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.negative_edge_cycle.html) 

# In[9]:


dg_order_book = order_book_to_dg(order_book)
nx.negative_edge_cycle(dg_order_book, weight="weight", heuristic=True)


# ### Finding and displaying one negative cycle
# 
# The NetworkX library includes the function [`find_negative_cycle` that locates a single negative edge cycle](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.find_negative_cycle.html) if one exists. We can use this to demonstrate the existence of an arbitrage opportunity, and to highlight that opportunity on the directed graph of all possible trades. The following cell reports the cycle found and the trading return measured in basis points (1 bp = 0.01%) 
# 
# Note this may or may not the trading cycle with maximum return. There may be other cycles with higher or lower returns, and that allow higher or lower trading volumes.

# In[10]:


# compute the sum of weights given a list of nodes
def sum_weights(cycle):
    return sum([dg_order_book.edges[edge]["weight"] for edge in zip(cycle, cycle[1:] + cycle[:1])])

dg_order_book = order_book_to_dg(order_book)
arb = nx.find_negative_cycle(dg_order_book, weight="weight", source="USD")[:-1]
print(f"Trading cycle: {arb}")
bp = 10000 * (np.exp(-sum_weights(arb)) - 1)
print(f"{bp:0.3f} basis points")
    
for src, dst in zip(arb, arb[1:] + arb[:1]):
    dg_order_book[src][dst]["width"] = 5
    
ax = draw_dg(dg_order_book, 0.05)
ax.set_title(f"Candidate Trading Cycle {bp:0.3f} basis points return")


# A brute force search over all cycles has complexity $(n + e)(c + 1)$ where $n$ is number of nodes, $e$ is the number of edges, and $c$ is the number of cycles. lkdlwhich is impractical for larger scale applications. A more efficient search based on the Bellman-Ford algorithm is embedded in the NetworkX function [`negative_edge_cycle`](https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.shortest_paths.weighted.negative_edge_cycle.html) that returns a logical True if a negative cycle exists in a directed graph. 

# Then if the shortest path algorithm returns a closed path with negative length, then that path will yield a positive financial return. Designating the source and destination as nodes $i_{0} i_{n}$, the largest return is given by a path minimizing the sum of weights computed as
# 
# $$
# \begin{align*}
# & \min\; W_k \\
# & \text{s.t.}\; W = \sum_{k=0}^{n-1} w_{i_k \rightarrow i_{k+1}}\\
# \end{align*}
# $$
# 
# An arbitrage exists if path length $W < 0$ for any path where $i_0 = i_n$.

# ## Brute force search arbitrage with simple cycles
# 
# A brute force search over for all simple cycles has order $(N_{nodes} + N_{edges})(N_{cycles} + 1)$ complexity, which is prohibitive for large order books. Nevertheless, we explore this option here to better understand the problem of finding and valuing arbitrage opportunities.
# 
# Here we compute the financial return for all simple cycles that can be constructed within a directed graph. The following cell uses [`simple_cycles`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html) from the NetworkX library to construct a dictionary of all distinct simple cycles in the order book. Each cycle is represented by an ordered list of nodes. For each cycle, the financial return is computed, and a histogram is constructed to show the distribution of potential returns. Several paths with the highest return are then overlaid on the graph of the order book.
# 
# Again, note that no account is taken of the trading capacity available on each path.

# In[11]:


# This cell iterates over all simple cycles in a directed graph. This
# can a long time for a large, well connected graph. 

# convert order book to a directed graph
dg_order_book = order_book_to_dg(order_book)

# compute the sum of weights given a list of nodes
def sum_weights(cycle):
    return sum([dg_order_book.edges[edge]["weight"] for edge in zip(cycle, cycle[1:] + cycle[:1])])

# create a dictionary of all simple cycles and sum of weights
cycles = {tuple(cycle): 10000 * (np.exp(-sum_weights(cycle)) - 1) for cycle in nx.simple_cycles(dg_order_book)}

print(f"There are {len(cycles)} distinct simple cycles in the order book.")
print(f"   {len([cycle for cycle in cycles if cycles[cycle] > 0])} of the cycles have positive return.")

# create histogram
plt.hist(cycles.values(), bins=int(np.sqrt(len(cycles))))
ax = plt.gca()
ax.set_ylabel("count")
ax.set_xlabel("Basis Points")
ax.set_title("Histogram of Returns for all Simple Cycles")
ax.grid(True)
ax.axvline(0, color='r')


# In[12]:


arbitrage = [cycle for cycle in sorted(cycles, key=cycles.get, reverse=True) if cycles[cycle] > 0]

n_cycles_to_list = 5

print(f"Top {n_cycles_to_list}\n")
print(f"Basis Points             Arbitrage Cycle")
for cycle in arbitrage[0: min(n_cycles_to_list, len(arbitrage))]:
    t = list(cycle)
    t.append(cycle[0])
    print(f"{cycles[cycle]:6.3f}         {len(t)} trades: {' -> '.join(t)}")


# In[13]:


n_cycles_to_show = 1

for cycle in arbitrage[0: min(n_cycles_to_show, len(arbitrage))]:

    # get fresh graph to color nodes
    dg_order_book = order_book_to_dg(order_book)
    
    # color nodes red
    for node in cycle:
        dg_order_book.nodes[node]['color'] = 'red'
    
    # makes lines wide
    for edge in zip(cycle, cycle[1:] + cycle[:1]):
        dg_order_book.edges[edge]['width'] = 4
        
    ax = draw_dg(dg_order_book, rad=0.05)

    t = list(cycle)
    t.append(cycle[0])
    ax.set_title(f" {len(t)} trades: {' -> '.join(t)}\n\n Return = {cycles[cycle]:6.3f} basis points ")


# ## Pyomo Model for Arbitrage with Capacity Constraints
# 
# The preceding analysis shows that depending on generic network algorithms to find arbitrage in an exchange order book has some practical limitations:
# 
# * Since there can be multiple negative cycles, there may be more than one opportunity for arbitrage. Which opportunity offers the greatest potential for profit?
# * Shortest path algorithms do not account for capacity constraints. The network algorithms are restricted to investments small enough to satisfy all capacity constraints on every edge.
# 
# Let's consider a Pyomo optimization model to find arbitrage opportunities that maximize financial return. We assume we are given a directed graph where each edge $i\rightarrow j$ is labeled with a 'multiplier' $a_{i\rightarrow j}$ indicating how many units of currency $j$ will be received for one unit of currency $i$, and a 'capacity' $c_{i\rightarrow j}$ indicating how many units of currency $i$ can be converted to currency $j$.
# 
# The model provides up to $T$ events for a given order book. Each event may consist of zero or currency transactions on different edges. A single transaction converts $x_{i\rightarrow j}(t)$ units of currency $i$ to currency $j$. Following the all transactions at event $t$, the trader will hold $v_j(t)$ units of currency $j$ where
# 
# $$v_{j}(t) = v_{j}(t-1) + \sum_{i\in I_j} a_{i\rightarrow j}x_{i\rightarrow j}(t) - \sum_{k\in O_j} x_{j\rightarrow k}(t)$$
# 
# Set $I_j$ are the input nodes to $j$, and set $O_j$ is the set of output nodes from $j$. Each edge on the order book has a finite capacity. For every edge $i\rightarrow j$, the sum of all transactions must satisfy
# 
# $$\sum_{t=1}^T x_{j\rightarrow k}(t) \leq c_{j\rightarrow k}$$
# 
# 
# The objective of the optimization model is to find a sequence of currency transactions the increase holdings of a reference currency. The solution is constrained by assuming the trader cannot short sell any currency. The resulting model is given by
# 
# $$
# \begin{align*}
# \max \quad & v_{USD}(T) \\
# \\
# \text{s.t.} \quad & v_{USD}(0) = v_0 \\ 
# \\
# & v_{j}(t) = v_{j}(t-1) + \sum_{i\in I_j} a_{i\rightarrow j}x_{i\rightarrow j}(t) - \sum_{k\in O_k} x_{j\rightarrow k}(t) & \forall j\in NODES, t=1, 2, \ldots, T \\
# & v_j(t-1) \geq \sum_{k\in O_j} x_{j\rightarrow k}(t) & \forall j\in NODES, t = 1, 2, \ldots, T  \\
# & \sum_{t=1}^T x_{j\rightarrow k}(t) \leq c_{j\rightarrow k} & \forall (j, k) \in EDGES,  t = 1, 2, \ldots, T \\
# \end{align*}
# $$
# 
# The function `crypto_model` creates an instance of a corresponding Pyomo model.

# In[14]:


import pyomo.environ as pyo

def crypto_model(dg_order_book, T = 10, v0 = 100.0):

    m = pyo.ConcreteModel(f"{exchange} arbitrage")

    # length of the trading chain
    m.T0 = pyo.RangeSet(0, T)
    m.T1 = pyo.RangeSet(1, T)

    # currency nodes and trading edges
    m.NODES = pyo.Set(initialize=list(dg_order_book.nodes))
    m.EDGES = pyo.Set(initialize=list(dg_order_book.edges))

    # currency on hand at each node
    m.v = pyo.Var(m.NODES, m.T0, domain=pyo.NonNegativeReals)

    # amount traded on each edge at each trade
    m.x = pyo.Var(m.EDGES, m.T1, domain=pyo.NonNegativeReals)

    # total amount traded on each edge over all trades
    m.z = pyo.Var(m.EDGES, domain=pyo.NonNegativeReals)

    # "multiplier" on each trading edge
    @m.Param(m.EDGES)
    def a(m, src, dst):
        return dg_order_book.edges[(src, dst)]["a"]

    @m.Param(m.EDGES)
    def c(m, src, dst):
        return dg_order_book.edges[(src, dst)]["capacity"]

    @m.Objective(sense=pyo.maximize)
    def wealth(m):
        return m.v["USD", T]

    @m.Constraint(m.EDGES)
    def total_traded(m, src, dst):
        return m.z[src, dst] == sum([m.x[src, dst, t] for t in m.T1])

    @m.Constraint(m.EDGES)
    def edge_capacity(m, src, dst):
        return m.z[src, dst] <= m.c[src, dst]

    # initial assignment of 100 units on a selected currency
    @m.Constraint(m.NODES)
    def initial(m, node):
        if node == "USD":
            return m.v[node, 0] == v0
        return m.v[node, 0] == 0.0

    @m.Constraint(m.NODES, m.T1)
    def no_shorting(m, node, t):
        out_nodes = [dst for src, dst in m.EDGES if src == node]
        return m.v[node, t - 1] >= sum(m.x[node, dst, t] for dst in out_nodes)

    @m.Constraint(m.NODES, m.T1)
    def balances(m, node, t):
        in_nodes = [src for src, dst in m.EDGES if dst == node]
        out_nodes = [dst for src, dst in m.EDGES if src == node]
        return m.v[node, t] == m.v[node, t - 1] + sum(m.a[src, node] * m.x[src, node, t] for src in in_nodes) - sum(m.x[node, dst, t] for dst in out_nodes) 

    solver = pyo.SolverFactory("cbc")
    solver.solve(m)
    
    return m


# Computing the potential return from an order book.

# In[15]:


dg_order_book = order_book_to_dg(order_book)

v0 = 10000.0
T = 8
m = crypto_model(dg_order_book, T=T, v0=v0)
vT = m.wealth()

print(f"Starting wealth = {v0:0.2f} USD")
print(f"Weath after {T:2d} transactions = {vT:0.2f} USD")
print(f"Return = {10000 * (vT - v0)/v0:0.3f} basis points")
print()


# In[16]:


for node in m.NODES:
    print(f"{node:5s}", end="")
    for t in m.T0:
        print(f" {m.v[node, t]():11.5f}", end="")
    print()


# In[17]:


print("\nTransaction Events")
for t in m.T1:
    print(f"t = {t}")
    for src, dst in m.EDGES:
        if m.x[src, dst, t]() > 1e-6:
            print(f" {src:8s} -> {dst:8s}: {m.x[src, dst, t]():14.6f} {m.a[src, dst] * m.x[src, dst, t]():14.6f}")
    print()


# In[18]:


# report what orders to issue
for src, dst in m.EDGES:
    if m.z[src, dst]() > 0.0000002: 
        dg_order_book.nodes[src]["color"] = "red"
        dg_order_book.nodes[dst]["color"] = "red"
        dg_order_book[src][dst]["width"] = 4
                
draw_dg(dg_order_book, 0.05)


# In[19]:


# report what orders to issue
print("Trading Summary for the Order Book")
print(f"  Order Book   Type    Capacity         Traded")
for src, dst in m.EDGES:
    if m.z[src, dst]() > 0.0000002:
        kind = dg_order_book.edges[(src,dst)]['kind']
        s = f"{src:>5s} -> {dst:<5s} {kind} {m.c[src, dst]:12.5f} {m.z[src, dst]():14.5f}"
        s += "  >>>  "
        if kind == "ask":
            base = dst
            quote = src
            symbol = base + "/" + quote
            price = 1.0 / dg_order_book.edges[(src, dst)]["a"]
            volume = m.z[src, dst]() / price
            s += f"sell {volume:15.6f} {symbol:11s} at {price:12.6f}"
            
        if kind == "bid":
            base = src
            quote = dst
            symbol = base + "/" + quote
            price = dg_order_book.edges[(src, dst)]["a"]
            volume = m.z[src,dst]() 
            s += f"buy {volume:16.6f} {symbol:11s} at {price:12.6f}"  
        print(s)
            


# In[20]:


print("\nTransaction Events")
for t in m.T1:
    print(f"t = {t}")
    for src, dst in m.EDGES:
        if m.x[src, dst, t]() > 0.0000002:
            print(f"{src:8s} -> {dst:8s}: {m.x[src, dst, t]():14.6f}")
    print()


# In[21]:


# display currency balances
balances = pd.DataFrame()
for node in dg_order_book.nodes:
    if sum(m.v[node, t]() for t in m.T0) > 0.0000002:
        for t in m.T0:
            balances.loc[t, node] = m.v[node, t]()

balances.plot(kind="bar", subplots=True, figsize=(8, 10), xlabel="Transaction", ylabel="Currency Units")
plt.gcf().tight_layout()
plt.show()


# ## Real Time Downloads of Order Books from an Exchange
# 
# The goal of this notebook is to show how network algorithms and optimization can be utilized to detect arbitrage opportunities within an order book that has been obtained from a cryptocurrency exchange.
# 
# The subsequent cell in the notebook utilizes `ccxt.exchange.fetch_order_book` to obtain the highest bid and lowest ask orders from an exchange for market symbols that meet the criteria of having a minimum in-degree for their base currencies.

# In[22]:


import pandas as pd

def get_order_book(exchange, dg):

    def get_orders(base, quote, limit=1):
        """
        Return order book data for a specified symbol.
        """
        result = exchange.fetch_order_book("/".join([base, quote]), limit)
        if not result["asks"] or not result["bids"]:
            result = None
        else:
            result["base"], result["quote"] = base, quote
            result["timestamp"] = exchange.milliseconds()
            result["bid_price"], result["bid_volume"] = result["bids"][0] 
            result["ask_price"], result["ask_volume"] = result["asks"][0]
        return result

    # fetch order book data and store in a dictionary
    order_book = filter(lambda r: r is not None, [get_orders(base, quote) for quote, base in dg.edges()])

    # convert to pandas dataframe
    order_book = pd.DataFrame(order_book)
    order_book["timestamp"] = pd.to_datetime(order_book["timestamp"], unit="ms")

    return order_book[['symbol', 'timestamp', 'base', 'quote', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume']]


minimum_in_degree = 5

# graph of market symbols with mininum_in_degree for base currencies
dg = get_dg(exchange, minimum_in_degree)

# retrieve order book for all markets in the graph
order_book = get_order_book(exchange, dg)

# find trades
v0 = 10000.0
m = crypto_model(dg_order_book, T=12, v0=v0)
wT = m.wealth()

print(f"Potential Return = {10000*(vT - v0)/v0:0.3f} basis points")
display(order_book)


# The following cell can be used to download additional order book data sets for testing.

# In[23]:


from datetime import datetime
import time
import glob

search_time = 20
timeout = time.time() + search_time

# wait for arbitrage opportunity
while time.time() <= timeout:
    print(".", end="")
    order_book = get_order_book(exchange, dg)
    dg_order_book = order_book_to_dg(order_book)
    if nx.negative_edge_cycle(dg_order_book, weight="weight", heuristic=True):
        print("arbitrage found!")
        fname = f"{exchange} orderbook {datetime.utcnow().strftime('%Y%m%d_%H_%M_%S')}.csv".replace(" ", "_")
        order_book.to_csv(fname)
        print(f"order book saved to: {fname}")
        break
else:
    print("no arbitrage found in {search_time} seconds")
    


# ## Bibliographic Notes
# 
# Crytocurrency markets are relatively new compared to other markets, and relatively few academic papers are available that specifically address arbitrage on those markets. Early studies, such as the following, reported periods of large, recurrent arbitrage opportunities that exist across exchanges, and that can persist for several days or weeks.
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

# In[ ]:




