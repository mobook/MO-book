#!/usr/bin/env python
# coding: utf-8

# <!--NOTEBOOK_HEADER-->
# *This notebook contains course material from [CBE40455](https://jckantor.github.io/CBE40455) by
# Jeffrey Kantor (jeff at nd.edu); the content is available [on Github](https://github.com/jckantor/CBE40455.git).
# The text is released under the [CC-BY-NC-ND-4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode),
# and code is released under the [MIT license](https://opensource.org/licenses/MIT).*

# <!--NAVIGATION-->
# < [Optimization under Uncertainty](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/06.00-Optimization-under-Uncertainty.ipynb) | [Contents](toc.ipynb) | [Scenario Analysis for a Plant Expansion](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/06.02-Scenario-Analysis-for-a-Plant-Expansion.ipynb) ><p><a href="https://colab.research.google.com/github/jckantor/CBE40455/blob/master/notebooks/06.01-Newsvendor-Problem.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://raw.githubusercontent.com/jckantor/CBE40455/master/notebooks/06.01-Newsvendor-Problem.ipynb"><img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>

# # Newsvendor Problem

# This notebook demonstrates the formulation and solution of the well-known "Newsvendor Problem" using GLPK/Mathprog.

# ## Background

# <p>
#     The newsvendor problem is a two stage decision problem with recourse. The 
#     vendor needs to decide how much inventory to order today to fulfill an 
#     uncertain demand. The data includes the unit cost, price, and salvage value of 
#     the product being sold, and a probabilistic forecast of demand. The objective 
#     is to maximize expected profit.
# </p>
# <p>
#     As shown in lecture, this problem can be solved with a plot, and the solution
#     interpreted in terms of a cumulative probability distribution. The advantage
#     of a MathProg model is that additional constraints or other criteria may be 
#     considered, such as risk aversion.</p>
# <p>
#     There is an extensive literature on the newsvendor problem which has been 
#     studied since at least 1888. See 
#     <a rel="external" href="http://www.isye.umn.edu/courses/ie5551/additional%20materials/newsvendort.pdf">here</a>
#    for a thorough discussion.
# </p>
# 

# ## Solution

# In[1]:


get_ipython().run_cell_magic('script', 'glpsol -m /dev/stdin', '\n# Example: Newsvendor.mod\n\n/* Unit Price Data */\nparam r >= 0;                              # Price\nparam c >= 0;                              # Cost\nparam w >= 0;                              # Salvage value\n\n/* Price data makes sense only if  Price > Cost > Salvage */\ncheck: c <= r;\ncheck: w <= c;\n\n/* Probabilistic Demand Forecast */\nset SCENS;                                 # Scenarios\nparam D{SCENS} >= 0;                       # Demand\nparam Pr{SCENS} >= 0;                      # Probability\n\n/* Probabilities must sum to one. */\ncheck: sum{k in SCENS} Pr[k] = 1;\n\n/* Expected Demand */\nparam ExD := sum{k in SCENS} Pr[k]*D[k];\n\n/* Lower Bound on Profit: Expected Value of the Mean Solution */\nparam EVM := -c*ExD + sum{k in SCENS} Pr[k]*(r*min(ExD,D[k])+w*max(ExD-D[k],0));\n\n/* Upper Bound on Profit: Expected Value with Perfect Information */\nparam EVPI := sum{k in SCENS} Pr[k]*(r-c)*D[k];\n\n/* Two Stage Stochastic Programming */\nvar x >= 0;                     # Stage 1 (Here and Now): Order Quqntity\nvar y{SCENS}>= 0;               # Stage 2 (Scenario Dep): Actual Sales\nvar ExProfit;                   # Expected Profit\n\n/* Maximize Expected Profit */\nmaximize OBJ: ExProfit;\n\n/* Goods sold are limited by the order quantities and the demand  */\ns.t. PRFT: ExProfit = -c*x + sum{k in SCENS} Pr[k]*(r*y[k] + w*(x-y[k]));\ns.t. SUPL {k in SCENS}: y[k] <= x;\ns.t. DMND {k in SCENS}: y[k] <= D[k];\n\nsolve;\n\ntable Table_EVM {k in SCENS} OUT "CSV" "evm.csv" "Table":\n   k~Scenario,\n   Pr[k]~Probability, \n   D[k]~Demand, \n   ExD~Order, \n   min(ExD,D[k])~Sold,\n   max(ExD-D[k],0)~Salvage, \n   -c*ExD + r*min(ExD,D[k]) + w*max(ExD-D[k],0)~Profit;\n   \ntable Table_EVPI {k in SCENS} OUT "CSV" "evpi.csv" "Table":\n   k~Scenario,\n   Pr[k]~Probability, \n   D[k]~Demand, \n   D[k]~Order, \n   D[k]~Sold,\n   0~Salvage, \n   -c*D[k] + r*D[k]~Profit;\n   \ntable Table_SP {k in SCENS} OUT "CSV" "evsp.csv" "Table":\n   k~Scenario,\n   Pr[k]~Probability, \n   D[k]~Demand, \n   x~Order, \n   y[k]~Sold,\n   x-y[k]~Salvage, \n   -c*x + r*y[k] + w*(x-y[k])~Profit;\n\ndata;\n\n/* Problem Data corresponds to a hypothetical case of selling programs prior \nto a home football game. */\n\nparam r := 10.00;                         # Unit Price\nparam c :=  6.00;                         # Unit Cost\nparam w :=  2.00;                         # Unit Salvage Value\n\nparam: SCENS:  Pr    D   :=\n       HiDmd   0.25  250\n       MiDmd   0.50  125\n       LoDmd   0.25   75 ;\n\nend;')


# ### Expected Value for the Mean Scenario (EVM)

# In[2]:


import pandas
evm = pandas.read_csv("evm.csv")
display(evm)

ev_evm = sum(evm['Probability']*evm['Profit'])
print "Expected Value for the Mean Scenario = {:6.2f}".format(ev_evm)


# ### Expected Value with Perfect Information (EVPI)

# In[3]:


evpi = pandas.read_csv("evpi.csv")
display(evpi)

ev_evpi = sum(evpi['Probability']*evpi['Profit'])
print "Expected Value with Perfect Information = {:6.2f}".format(ev_evpi)


# ### Expected Value by Stochastic Programming

# In[4]:


evsp = pandas.read_csv("evsp.csv")
display(evsp)

ev_evsp = sum(evsp['Probability']*evsp['Profit'])
print "Expected Value by Stochastic Programming = {:6.2f}".format(ev_evsp)


# ### Value of Perfect Information

# In[5]:


print "Value of Perfect Information = {:6.2f}".format(ev_evpi-ev_evsp)


# ### Value of the Stochastic Solution

# In[6]:


print "Value of the Stochastic Solution = {:6.2f}".format(ev_evsp-ev_evm)


# In[6]:





# In[30]:


r = 1.00
c = 0.60
w = 0.25

def profit(D,x):
    return r*min([D,x]) + w*max([0,x-D]) - c*x


# In[31]:


scenarios = [['Low Demand',75,.25],['High Demand',200,.75]]


# In[33]:


def exprofit(x):
    v = 0
    for s in scenarios:
        v += s[2]*profit(s[1],x)
    return profit

x = linspace(0,400,400)
exprofit(100)


# In[23]:


x = linspace(0,400,400)
plot(x,map(exprofit,x))
xlabel('Order size')
ylabel('Expected Profit')


# In[ ]:





# <!--NAVIGATION-->
# < [Optimization under Uncertainty](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/06.00-Optimization-under-Uncertainty.ipynb) | [Contents](toc.ipynb) | [Scenario Analysis for a Plant Expansion](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/06.02-Scenario-Analysis-for-a-Plant-Expansion.ipynb) ><p><a href="https://colab.research.google.com/github/jckantor/CBE40455/blob/master/notebooks/06.01-Newsvendor-Problem.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://raw.githubusercontent.com/jckantor/CBE40455/master/notebooks/06.01-Newsvendor-Problem.ipynb"><img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>
