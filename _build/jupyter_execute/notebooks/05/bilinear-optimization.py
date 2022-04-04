#!/usr/bin/env python
# coding: utf-8

# # Some Bilinear Optimization Examples

# ## Example: Pooling Sources for a Milk Wholesale Distributor
# 
# A small distributor of wholesale milk creates custom blends of raw milk supplied from farms for delivery to customers. The distributor has found an opportunity to improve profits by buying raw milk from farms located at a long distance away key customers. The farmer has only one truck to tranport raw milk from the distant farms. How should the distributor pool raw milk from the distant farms in order to increase profits? 

# In[64]:


import pandas as pd

raw_milk_suppliers = pd.DataFrame({
    "Farm A": {"fat": 0.045, "price": 45.0, "location": "local"},
    "Farm B": {"fat": 0.030, "price": 42.0, "location": "local"},
    "Farm C": {"fat": 0.033, "price": 32.0, "location": "remote"},
    "Farm D": {"fat": 0.050, "price": 40.0, "location": "remote"}
}).T

customers = pd.DataFrame({
    "Customer A": {"fat": 0.040, "price": 52.0, "demand": 1000.0},
    "Customer B": {"fat": 0.030, "price": 48.0, "demand": 5000.0}
}).T

display(raw_milk_suppliers)
display(customers)


# * Given the available supplies, both local and remote, what is the maximum profit the distributor can earn assuming there was no necessity to mix milk from the remote suppliers? Given that solution, would there be any problem with pooling the milk from the remote farms prior to delivery? 
# 
# * Reformulate the problem to help the distributor maximize profits by pooling supplies from the remote farms.

# In[66]:


import pyomo.environ as pyo

m = pyo.ConcreteModel()

m.S = pyo.Set(initialize=raw_milk_suppliers.index)
m.C = pyo.Set(initialize=customers.index)
m.x = pyo.Var(m.S, m.C, domain=pyo.NonNegativeReals)

@m.Param(m.S)
def cost(m, s):
    return raw_milk_suppliers.loc[s, "price"]

@m.Param(m.S)
def fat(m, s):
    return raw_milk_suppliers.loc[s, "fat"]

@m.Param(m.C)
def price(m, c):
    return customers.loc[c, "price"]

@m.Param(m.C)
def fat_spec(m, c):
    return customers.loc[c, "fat"]

@m.Param(m.C)
def demand(m, c):
    return customers.loc[c, "demand"]

@m.Objective(sense=pyo.maximize)
def profit(m):
    return sum(m.x[s, c]*(m.price[c] - m.cost[s]) for s, c in m.S * m.C)

@m.Constraint(m.C)
def demand_limit(m, c):
    return sum(m.x[s, c] for s in m.S) <= m.demand[c]

@m.Expression(m.S, m.C)
def fat_shipped(m, s, c):
    return m.x[s, c]*m.fat[s]

@m.Constraint(m.C)
def fat_constraint(m, c):
    return  sum(m.fat_shipped[s, c] for s in m.S) >= sum(m.x[s, c]*m.fat_spec[c] for s in m.S)

pyo.SolverFactory('cbc').solve(m)

print(f"{m.profit():0.2f}")

soln = pd.DataFrame([[s, 
                      c, 
                      m.x[s, c](), 
                      raw_milk_suppliers.loc[s, "location"],
                      round(m.fat_shipped[s, c](), 0)
                     ] for s, c in m.S * m.C],
                    columns = ["supplier", "customer", "shipped", "location", "fat_shipped"])

display(soln)

pd.pivot_table(soln, index="supplier", columns=["customer"])


# $$
# \begin{align*}
# \sum_{}u_{s, p} & = \sum_{}v_{p, c} & \forall p \in \text{POOLS}
# \end{align*}
# $$

# ### Splitting a stream
# 
# Consider a stream from a pool with flowrate $P$ and compositions $\{y_i\}_{i=1}^n$. The stream is split into multiple streams $P_j$, $j=1, \dots, J$. The material balances are written
# 
# $$P y_i = \sum_{j=1}^J y_{i,j} P_j$$
# 
# Dividing by $P$ and letting $\phi_j = P_j/P$
# 
# $$y_i = \sum_{j=1}^J y_{i,j}\phi_j$$
# 
# where $\phi_j$ is the split ratio and $\sum_{j=1}^J \phi_1 = 1$.

# ## Example: Heat Engines
# 
# Endoreversible thermodynamics is subfield of irreversible thermodynamics focusing on the fundamental analysis of energy conversion accounting for finite resistances to heat transfer. Classical thermodynamics studies the theoretical limit of engines without the heat transfer resistances that limit performance in real world applications. It has been found that the efficiency bounds developed in this manner are generally closer estimates of actual performance than the more familiar Carnot efficiency described in introductory textbooks. The example described here is generally attributed to Curzon and Ahlborn (1975).
# 
# >Curzon, F. L., & Ahlborn, B. (1975). Efficiency of a Carnot engine at maximum power output. American Journal of Physics, 43(1), 22-24. 
# 
# The following paper applies these concepts to the optimal design of a geothermally powered air conditioning system. This paper would provide a good didactic example for the solution of a bilinear optimization problem.
# 
# > Davis, G. W., & Wu, C. (1997). Finite time analysis of a geothermal heat engine driven air conditioning system. Energy conversion and management, 38(3), 263-268.
# 
# Given a hot and cold reservoir at temperature $T_h$ and $T_c$, respectively, the mechanical power $\dot{w}$ that is extracted by any machine is subject to the energy balances.
# 
# $$
# \begin{align*}
# \dot{w} & = q_h - q_c & \text{power production}\\
# \dot{q}_h & = U_h (T_h - T_h') & \text{heat transfer - hot} \\
# \dot{q}_c & = U_c (T_c' - T_c) & \text{heat transfer - cold}\\
# \end{align*}
# $$
# 
# The heat transfer coefficients $U_h$ and $U_c$ model the transmission of heat to and from the engine that is converting heat to work. Heat is released from the hot reservoir at temperature $T_h'$ thereby increasing increasing entropy of the surroundings at a rate ${\dot{q}_h}/{T_h'}$. The heat absorbed at the cold temperature reservoir is decreasing entropy of the surroundings at a rate ${\dot{q}_c}/{T_c'}$. From the second law,
# 
# $$
# \begin{align*}
# \frac{\dot{q}_h}{T_h'} < \frac{\dot{q}_c}{T_c'} \\
# \end{align*}
# $$
# 
# Defining $\dot{\sigma}_h = \dot{q}_h/T_h'$ and $\dot{\sigma}_c = \dot{q}_c/T_c'$, and given data for the hot and cold utilities, $T_h$, $T_c$, $U_h$ and $U_c$, the problem is find the maximum mechanical power $\dot{w}$ subject to the constraints
# 
# $$
# \begin{align*}
# & \max\ \dot{w} \\
# \\
# \text{s.t.}\qquad\qquad
# \dot{w} & = \dot{q}_h - \dot{q}_c & \text{power production}\\
# \dot{q}_h & = U_h (T_h - T_h') & \text{heat transfer - hot} \\
# \dot{q}_c & = U_c (T_c' - T_c) & \text{heat transfer - cold}\\
# \dot{q}_h & = \dot{\sigma}_h T_h' \\
# \dot{q}_c & = \dot{\sigma}_c T_c' \\
# \dot{\sigma}_h & < \dot{\sigma}_c &
# \end{align*}
# $$
# 
# The notation $\dot{\sigma}$ is used in place of entropy in order to formulate the model with non-negative variables.

# In[37]:


import pyomo.kernel as pmo

Th = 500
Tc = 300

Uh = 3
Uc = 3

m = pmo.block()

m.Th = pmo.variable(lb=0)
m.Tc = pmo.variable(lb=0)
m.qh = pmo.variable(lb=0)
m.qc = pmo.variable(lb=0)
m.sh = pmo.variable(lb=0)
m.sc = pmo.variable(lb=0)
m.w = pmo.variable()

m.energy_balance = pmo.constraint(m.w == m.qh - m.qc)
m.heat_source = pmo.constraint(m.qh == Uh*(Th - m.Th))
m.heat_sink = pmo.constraint(m.qc == Uc*(m.Tc - Tc))
m.entropy_out = pmo.constraint(m.qh == m.sh * m.Th)
m.entropy_in = pmo.constraint(m.qc == m.sc * m.Tc)
m.second_law = pmo.constraint(m.sh <= m.sc)

m.objective = pmo.objective(-m.w)

pmo.SolverFactory('ipopt').solve(m)

print(f"maximum work = {m.w()}")

print("Th = ", m.Th())
print("Tc = ", m.Tc())
print("qh = ", m.qh())
print("qc = ", m.qc())
print("sh = ", m.sh())
print("sc = ", m.sc())


# ## Example: Isothermal Flash Calculation
# 
# https://pubs.acs.org/doi/pdf/10.1021/ie300183e
# 
# Flash calculations are an essential element of models with vapor-liquid equilibrium, including chemical process design, reservoir simulations, environmental modeling.
# 
# A feed stream $F$ with composition mole fractions $\{z_i\}_{i=1}^n$ is split into a vapor stream $V$ with composition mole fractions $\{y_i\}_{i=1}^n$, and a liquid stream $L$ with composition mole fractions $\{x_i\}_{i=1}^n$. At equilibrium the vapor and liquid compositions satisfy a relationship
# 
# $$
# \begin{align*}
# y_i & = K_i x_i & \forall i = 1, \dots, n
# \end{align*}
# $$
# 
# A material balance is written for each component
# 
# $$
# \begin{align*}
# F z_i & = L x_i + V y_i & \forall i = 1, \dots, n
# \end{align*}
# $$
# 
# The overall material balance $L = V - F$. Dividing by $F$ and letting $\phi = V/F$ results in $2n$ equations for $2n + 1$ variables $\phi$, $\{x_i\}_{i=1}^n$ and $\{y_i\}_{i=1}^n$
# 
# $$
# \begin{align*}
# z_i & = (1-\phi) x_i + \phi y_i & \forall i = 1, \dots, n \\
# y_i & = K_i x_i& \forall i = 1, \dots, n \\
# \end{align*}
# $$
# 
# The problem is fully defined with either $\sum_{i=1}^n x_i = 1$ or $\sum_{i=1}^n x_i = 1$.
# 

# ### McCormick relaxtions
# 
# $$x_i = \frac{z_i}{1 + (K_i - 1)\phi}$$
# 
# The isothermal flash model is given by
# 
# $$z_i = x_i + (K_i - 1)\phi x_i$$
# 
# where $0 \leq x_i \leq 1$ and $0 \leq \phi \leq 1$
# 
# Define $w_i = \phi x_i$
# 
# $$
# \begin{align*}
# w_i & \geq 0 \\
# w_i & \geq x - y - 1 \\
# w_i & \leq x \\
# w_i & \leq \phi
# \end{align*}
# $$

# In[55]:


import pyomo.environ as pyo
import numpy as np

# data
K = np.array([2.0, 1.3, 0.3])
z = np.array([0.3, 0.3, 0.4])

assert len(K) == len(z)
assert any([k > 1 for k in K])
assert any([k < 1 for k in K])
assert sum(z) == 1.0

m = pyo.ConcreteModel()
m.C = pyo.RangeSet(len(K))
m.x = pyo.Var(m.C, domain=pyo.NonNegativeReals)
m.y = pyo.Var(m.C, domain=pyo.NonNegativeReals)
m.f = pyo.Var(bounds=(0, 1))

@m.Param(m.C)
def K(m, c):
    return K[c-1]

@m.Param(m.C)
def z(m, c):
    return z[c-1]

@m.Constraint(m.C)
def VL_equil(m, c):
    return m.y[c] == m.K[c] * m.x[c]

@m.Constraint()
def y_mole_fractions(m):
    return sum(m.y[c] for c in m.C) == 1

@m.Constraint(m.C)
def mass_balance(m, c):
    return m.z[c] == m.x[c] + m.f*(m.y[c] - m.x[c])

pyo.SolverFactory('ipopt').solve(m)

print (m.f())

for c in m.C:
    print(f"{c}) K = {m.K[c]} z = {m.z[c]}  x = {m.x[c]():0.4f}  y = {m.y[c]():0.4f}")


# Relationship to a generalized eigenvalue problem.
# 
# $$
# \begin{align*}
# \left(
# \begin{bmatrix}
# 1 & 0 & \dots & 0 & -z_1 \\
# 0 & 1 & \dots & 0 & -z_2 \\
# \vdots & \vdots & \ddots & \vdots & \vdots \\
# 0 & 0 & \cdots & 1 & -z_n \\
# 1 & 1 & \cdots & 1 & - 1 \\
# \end{bmatrix} -
# \phi
# \begin{bmatrix}
# 1 - K_1 & 0 & \cdots & 0 & 0 \\
# 0 & 1 - K_2 & \cdots & 0 & 0 \\
# \vdots & \vdots & \ddots & \vdots & \vdots\\
# 0 & 0 & \cdots & 1 - K_n & 0 \\
# 0 & 0 & \cdots & 0 & 0
# \end{bmatrix}
# \right)
# \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n \\ 1 \end{bmatrix} 
# & = 
# \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 0 \end{bmatrix}
# \end{align*}
# $$
