#!/usr/bin/env python
# coding: utf-8

# # Crypto Currency Analysis
# 
# Crpytocurrency exchanges are websites that enable the purchase, sale, and exchange of cryptocurrencies. These exchanges provide liquidity for owners and establish the relative value of these currencies. As of this writing (mid-2022), [it is estimated](https://www.statista.com/statistics/730876/cryptocurrency-maket-value/) that cryptocurrencies have a collective market capitalization of more than 2 trillion USD. Cryptocurrency markets are constantly changing with new entrants, the occasional collapse of a currency, and highly volatile prices.
# 
# The purpose of this notebook is to explore the efficiency of cryptocurrency exchanges by testing for arbitrage opportunities. An arbitrage exists if a customer can realize a net profit through a sequence of risk-free trades. The efficient market hypothesis assumes arbitrage opportunities are quickly identified and exploited by investors. As a result of their trading, prices would reach a new equilibrium so that in an efficient market, any arbitrage opportunities would be small and fleeting. 
# 
# Still, the market has to get to equilibrium somehow. So it is possible, with real-time data and rapid execution, a trader can be in a position to profit from these fleeting arbitrage opportunities?

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

# ## Installations and Imports
# 
# This notebook requires multiple libraries. The following cell performs the required installations for Google Colab. To run in your own environment you will need to install `pyomo`,`ccxt`, and `networkx` python libraries, and a linear solver for Pyomo.

# In[156]:


import os
import sys
from time import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import networkx as ns
import numpy as np
import pandas as pd
import pyomo.environ as pyo

if "google.colab" in sys.modules:
    get_ipython().system('pip install -q ccxt')
    get_ipython().system('wget -N -q https://raw.githubusercontent.com/jckantor/MO-book/main/tools/install_on_colab.py ')
    get_ipython().run_line_magic('run', 'install_on_colab.py')


# ## Cryptocurrency Exchanges
# 
# Cryptocurrency exchanges are digital marketplaces for buying and trading cryptocurrencies. Joining an exchange enables a user to maintain multiple currencies in a digital wallet, buy and sell currencies, and to use cryptocurrencies in financial transactions. The [open-source library `ccxt`](https://github.com/ccxt/ccxt) currently supports real-time APIs for the largest and most common exchanges on which cryptocurrencies are traded. Here we import the library and list current exchanges supported by `ccxt`.

# In[167]:


import ccxt

for i, exchange in enumerate(ccxt.exchanges):
    print("") if not i % 4 else None
    print(f"{i+1:3d}) {exchange.ljust(20)}", end="")


# ## Representing an Exchange as a Directed Graph
# 
# Each exchange `ccxt` consists of multiple markets. Each market consists of all trading done between two specific currencies. `ccxt` labels each market with a symbol that is common across exchanges and suitable for within-exchange and cross-exchange arbitrage analyses.
# 
# The market symbol is an upper case string consisting of a slash (`/`) that separate abbreviations for a pair of traded currencies. The first abbreviation is for the base currency, the second is for the quote currency.  Prices for the base currency are denominated in units of the quote currency. As an example, the symbol `ETH/BTC` refers to a market for the base currency Ethereum (ETH) quoted in units of the Bitcoin(BTC).
# 
# A directed graph can be constructed from the market symbols available on a single exchange. Currencies are represented by nodes on the directed graph. Market symbols correspond to edges in the directed graph, with the source indicating the quote currency and the destination indicating the base currency.
# 
# The in-degree of a node refers to the number of incoming edges. Out-degree refers to the number of outgoing edges. Nodes with out-degrees greater than zero are highlighted because they represent currencies used to quote the price of other currencies. For all other nodes, a `minimum_in_degree` specifies a threshold value for in_degree for nodes to be displayed and retained for further analysis.
# 
# 

# In[168]:


# global variables used in subsequent cells
exchange = ccxt.binanceus()
markets = exchange.load_markets()
symbols = exchange.symbols


def symbols_to_dg(symbols, minimum_in_degree=1):

    # create an edge for every symbol
    dg = nx.DiGraph()
    for base, quote in [symbol.split("/") for symbol in symbols]:
        dg.add_edge(quote, base)

    remove_nodes = []
    for node in dg.nodes():
        if dg.out_degree(node) > 0:
            # color quote currencies in gold
            dg.nodes[node]["color"] = "gold"
        else:
            if dg.in_degree(node) < minimum_in_degree:
                # remove base currencies with an insufficint in_degree
                remove_nodes.append(node)
            else:
                # color base currencies in lightblue
                dg.nodes[node]["color"] = "lightblue"

    dg.remove_nodes_from(remove_nodes)
    return dg


def draw_dg(dg):

    fig = plt.figure(figsize=(15, 15))
    pos = nx.circular_layout(dg)
    nx.draw(
        dg,
        pos,
        with_labels=True,
        node_color=[dg.nodes[node]["color"] for node in dg.nodes()],
        node_size=1000,
        font_size=8,
        arrowsize=15,
        connectionstyle="arc3, rad=0.1",
    )
    #nx.draw_networkx_edge_labels(
    #    dg, pos, edge_labels={(src, dst): f"{src}/{dst}" for src, dst in dg.edges()}
    #)


dg_symbols = symbols_to_dg(symbols, 4)
draw_dg(dg_symbols)

print(f"Number of nodes = {len(dg_symbols.nodes()):3d}")
print(f"Number of edges = {len(dg_symbols.edges()):3d}")


# ## Exchange Order Book

# The order book for a currency exchange is the real-time inventory of trading orders. 
# 
# A **bid** is an order to buy up to a specified amount of the base currency at a price no greater than value specified in the quote currency. The exchange attempts to match the bid to a sell order at a price less than or equal to the bid price. If a transaction occurs, the  buyer will receive an amount of base currency less than or equal to the bid volume at a prices less than or equal to the bid price.
# 
# An **ask** is an offer to sell up to a specified amount of the base currency at a price no less than a value specified given in the quote currency. If a transaction occurs, then seller will sell no more than a specified about of the base currency at a price no less than the specified value if the exchange matches the ask order to a higher bid. 
# 
# The exchange order book maintains a list of all active orders for symbols traded on the exchange. Incoming bids above the lowest ask or incoming asks below the highest bid will be immediately matched and transactions executed following rules of the exchange. 
# 
# The following cell uses the `ccxt` library to fetch the highest bid and lowest ask from the order book for all trading symbols in a directed graph.

# In[169]:


import pandas as pd


def fetch_order_book(dg):

    # get trading symbols from exchange graph
    trade_symbols = ["/".join([base, quote]) for quote, base in dg.edges()]

    def fetch_order_book_symbol(symbol, limit=1, exchange=exchange):
        """return order book data for a specified symbol"""
        start_time = timer()
        result = exchange.fetch_order_book(symbol, limit)
        result["base"], result["quote"] = symbol.split("/")
        result["run_time (ms)"] = round(1000 * (timer() - start_time), 1)
        result["timestamp"] = exchange.milliseconds()
        if result["bids"]:
            result["bid_price"] = result["bids"][0][0]
            result["bid_volume"] = result["bids"][0][1]
        if result["asks"]:
            result["ask_price"] = result["asks"][0][0]
            result["ask_volume"] = result["asks"][0][1]
        return result

    # fetch order book data and store in a dictionary
    order_book = {symbol: fetch_order_book_symbol(symbol) for symbol in trade_symbols}

    # convert to pandas dataframe
    order_book = pd.DataFrame(order_book).T
    order_book.drop(columns=["datetime", "symbol"], inplace=True)
    order_book["timestamp"] = pd.to_datetime(order_book["timestamp"], unit="ms")

    return order_book


order_book = fetch_order_book(dg_symbols)
display(order_book)


# ## Order Book as a Directed Graph

# The order book can be organized as a directed graph.. 
# 
# Consider a market symbol with base currency $b$ and quote currency $q$ corresponding to a trading symbol $b/q$. A sell order for symbol $b/q$ presents an opportunity to purchase up to a specified amount of currency $b$ at the specified ask price. This sell order is represented on the directed graph by a directed edge from the quote currency to the base currency with labeled with a coefficient $a_{q\rightarrow b}$ equal to inverse of the ask price. The coefficient represents the conversion of one unit of the quote currency into $a_{q\rightarrow b}$ units of the base currency. The capacity of the edge, $c_{q\rightarrow b}$, is equal to the units of base currency available for trading in this sell order. The capacity is computed by multiplying the ask volume of the order book by the ask price.
# 
# A buy order for symbol $b/q$ is an opportunity to sell an amount of currency $b$ at the specified bid price. The is represented by a directed edge from the base currency to the quote currency with a coefficient value $a_{b\rightarrow q}$ equal to the bid price. Execution of the trade results each unit of the base currency being converted to $a_{b\rightarrow q}$ units of the quote currency. The capacity of the edge $c_{b\rightarrow q}$ is equal to bid volume.

# In[170]:


# dictionary of edges for (src, dst) tuples
#     type: 'bid' or 'ask'
#     a[src, dst]: 1 unit of src currency produces conv units of dst currency
#     log10_conv: log10 of conv


def order_book_to_dg(order_book):

    edges = dict()

    for symbol in order_book.index:
        if not np.isnan(order_book.at[symbol, "bid_volume"]):
            src = order_book.at[symbol, "base"]
            if src == "USD":
                src = "USD-SRC"
            dst = order_book.at[symbol, "quote"]
            if dst == "USD":
                dst = "USD-DST"
            edges[(src, dst)] = {
                "type": "bid",
                "a": order_book.at[symbol, "bid_price"],
                "log10_a": np.log10(order_book.at[symbol, "bid_price"]),
                "c": order_book.at[symbol, "bid_volume"],
            }
        if not np.isnan(order_book.at[symbol, "ask_volume"]):
            src = order_book.at[symbol, "quote"]
            if src == "USD":
                src = "USD-SRC"
            dst = order_book.at[symbol, "base"]
            if dst == "USD":
                dst = "USD-DST"
            edges[(src, dst)] = {
                "type": "ask",
                "a": 1.0 / order_book.at[symbol, "ask_price"],
                "log10_a": np.log10(1.0 / order_book.at[symbol, "ask_price"]),
                "c": order_book.at[symbol, "ask_volume"]* order_book.at[symbol, "ask_price"],
            }

    dg_order_book = nx.DiGraph()

    for (src, dst), data in edges.items():
        dg_order_book.add_edge(src, dst, weight=-data["log10_a"])

    for node in dg_order_book.nodes():
        dg_order_book.nodes[node]["color"] = "lightblue"

    return dg_order_book


dg_order_book = order_book_to_dg(order_book)

draw_dg(dg_order_book)


# ## Finding Arbitrage as Shortest Path
# 
# Given a directed graph representing the order book, the challenge is determine if there is an opportunity to begin with a fixed amount of a given currency, execute a series of trades resulting in an increase in the amount of that currency. Labeling each edge with a weight equal to the negative logarithm 
# 
# $$w_{i\rightarrow j} = - \log a_{i\rightarrow j}$$
# 
# and nodes $n_{SRC}$ and $n_{DST}$ (which may be the same node), the task is find series of nodes
# 
# $$n_{SRC} \rightarrow n_{i_1} \rightarrow n_{i_2} \rightarrow \cdots \rightarrow n_{i_J} \rightarrow n_{DST}$$
# 
# such that
# 
# $$w_{n_{SRC} \rightarrow n_{i_1}} + \sum_{j=1}^J w_{n_{i_j} \rightarrow n_{i_{j+1}}} +  w_{i_J \rightarrow DST} \leq 0$$

# In[171]:


dg_order_book = order_book_to_dg(order_book)
nx.negative_edge_cycle(dg_order_book, weight="weight")


def cycle_val(cycle):
    cycle.append(cycle[0])
    return sum([dg_order_book.edges[edge]["weight"] for edge in zip(cycle[0:-1], cycle[1:])])

w_min = np.inf
cycle_min = []
for cycle in nx.simple_cycles(dg_order_book):
    w = cycle_val(cycle)
    if w < 0:
        if w < w_min:
            w_min = w
            cycle_min = cycle

print(cycle_min, w_min, 100*(10**-w_min - 1), "%")


# In[172]:


def dg_shortest_path(dg_order_book):
    shortest_path = nx.shortest_path(
        dg_order_book, "USD-SRC", "USD-DST", weight="weight", method="bellman-ford"
    )

    S = 1.0
    x = np.inf
    for src, dst in zip(shortest_path, shortest_path[1:]):
        print(f"Sell {src}, Buy {dst}:")

        cap = edges[(src, dst)]["capacity"]
        a = edges[(src, dst)]["conv"]
        print(f"     Max {src} on-hand = {x}")
        print(f"     Max trade = {cap} {src}")

        a = edges[(src, dst)]["conv"]
        S = S * a
        x = a * min(cap, x)

        print(f"     Price = {a} {dst} per {src}")
        print(f"     Max position after trade {dst} = {x}")

        print()

    print("Summary")
    print(f"Starting Position {x/S} USD")
    print(f"Final Position {x} USD")
    print(f"Net profit = {x - x/S}  ({100*(1 - 1/S):.5f}%)")


dg_shortest_path(dg_order_book)


# In[ ]:


order_book = fetch_order_book(dg_symbols)
dg_order_book = order_book_to_dg(order_book)
dg_shortest_path(dg_order_book)


# ## Optimizing Wealth Creation

# In[555]:


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


# In[ ]:


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

