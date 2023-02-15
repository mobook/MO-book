#!/usr/bin/env python
# coding: utf-8

# # Extra material: Cryptarithms puzzle
# 
# :::{index} disjunctive programming
# :::
# 
# The July 1924 issue of the famous British magazine *The Strand* included a word puzzle by Henry E. Dudeney in his regular contribution "Perplexities". The puzzle is to assign a unique digit to each letter appearing in the equation
# 
#         S E N D
#       + M O R E
#     = M O N E Y
# 
# such that the arithmetic equation is satisfied, and the leading digit for M is non-zero. There are [many more examples](http://cryptarithms.awardspace.us/puzzles.html) of these puzzles, but this is perhaps the most well-known.
# 
# This notebook demonstrates a solution to this puzzle using Pyomo disjuctions and the `gecode` solver, a constraint solving package written in C++. This same puzzle is used in the `gecode` documentation, so this notebook may provide a useful contrast between Pyomo modeling and use of a native C++ API.

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_cbc()


# ## Modeling and Solution
# 
# There are several possible approaches to modeling this puzzle in Pyomo. 
# 
# [One approach](https://stackoverflow.com/questions/67456379/pyomo-model-constraint-programming-for-sendmore-money-task)  would be to using a matrix of binary variables $x_{a,d}$ indexed by letter $a$ and digit $d$ such that $x_{a,d} = 1$ designates the corresponding assignment. The problem constraints can then be implemented by summing the binary variables along the two axes. The arithmetic constraint becomes a more challenging.
# 
# [Another approach](https://www.gecode.org/doc/6.0.1/MPG.pdf) is to use Pyomo integer variables indexed by letters, then setup an linear expression to represent the puzzle. If we use the notation $n_a$ to represent the digit assigned to letter $a$, the algebraic constraint becomes
# 
# $$
# \begin{align*}
# 1000 n_s + 100 n_e + 10 n_n + n_d \\
# + 1000 n_m + 100 n_o + 10 n_r + n_e \\ 
# = 10000 n_m + 1000 n_o + 100 n_n + 10 n_e + n_y
# \end{align*}
# $$
# 
# The requirement that no two letters be assigned the same digit can be represented as a \ disjunction. Letting $n_a$ and $n_b$ denote the integers assigned to letters $a$ and $b$, the disjuction becomes
# 
# $$
# \begin{align*}
# \begin{bmatrix}n_a \lt n_b\end{bmatrix} 
# \ \veebar\  &
# \begin{bmatrix}n_b \lt n_a\end{bmatrix} 
# & \forall a \lt b
# \end{align*}$$
# 

# In[2]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

m = pyo.ConcreteModel()

m.LETTERS = pyo.Set(initialize=['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y'])
m.PAIRS = pyo.Set(initialize=m.LETTERS * m.LETTERS, filter = lambda m, a, b: a < b)
m.n = pyo.Var(m.LETTERS, domain=pyo.Integers, bounds=(0, 9))

@m.Constraint()
def message(m):
    return               1000*m.n['S'] + 100*m.n['E'] + 10*m.n['N'] + m.n['D'] \
                       + 1000*m.n['M'] + 100*m.n['O'] + 10*m.n['R'] + m.n['E'] \
     == 10000*m.n['M'] + 1000*m.n['O'] + 100*m.n['N'] + 10*m.n['E'] + m.n['Y']

# leading digit must be non-zero
@m.Constraint()
def leading_digit_nonzero(m):
    return m.n['M'] >= 1

# assign a different number to each letter
@m.Disjunction(m.PAIRS)
def unique_assignment(m, a, b):
    return [m.n[a] >= m.n[b] + 1, m.n[b] >= m.n[a] + 1]

# assign a "dummy" objective to avoid solver errors
@m.Objective()
def dummy_objective(m):
    return m.n['M']

pyo.TransformationFactory('gdp.bigm').apply_to(m)
solver = pyo.SolverFactory('cbc')
solver.solve(m)

def letters2num(s):
    return ' '.join(map(lambda s: f"{int(m.n[s]())}", list(s)))

print("    ", letters2num('SEND'))
print("  + ", letters2num('MORE'))
print("  ----------")
print("= ", letters2num('MONEY'))


# ## Suggested exercises
# 
# 1. Pyomo includes a logic-based solver `GDPopt` for generalized disjunctive programming problems. Implement and test `GDPopt` using combinations of solution strategies and MIP solvers. Compare the performance of `GDPopt` to the constraint solver `gecode`.
# 
# 2. There are [many more examples](http://cryptarithms.awardspace.us/puzzles.html) this puzzles. Refactor this code and create a function that can be used to solve generic puzzles of this type.

# In[ ]:




