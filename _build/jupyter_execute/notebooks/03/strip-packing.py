#!/usr/bin/env python
# coding: utf-8

# ```{index} disjunctive programming
# ```
# ```{index} boolean variables
# ```
# ```{index} single: Pyomo; boolean 
# ```
# 
# # Extra material: Strip packing
# 
# *Strip packing* (SP) refers to the problem of packing rectangles onto a two dimensional strip of fixed width. 
# 
# Many variants of this problem have been studied, the most basic is the pack a set of rectangles onto the shortest possible strip without rotation. Other variants allow rotation of the rectangles, require a packing that allows cutting the rectangles out of the strip with edge-to-edge cuts (guillotine packing), extends the strip to three dimensions, or extends the packing to non-rectangular shapes.
# 
# The extensive study of strip packing problems is motivated by their many industrial applications including
# 
# * placement of macro cells in semiconductor layouts,
# * wood and textile cutting operations,
# * laying out workstations in manufacturing facilities,
# * allocating communications bandwidth between two endpoints,
# * planning and scheduling $CO_2$ utilization for enhanced oil recovery,
# * scheduling allocations of a common resource.
# 
# Finding optimal solutions to strip packing problems is combinatorially difficult. Strip packing belongs to a class of problems called "NP-hard" for which known solution algorithms require effort that grows exponentially with problem size. For that reason much research on strip packing has been directed towards practical heuristic algorithms for finding good, though not optimal, for industrial applications.
# 
# Here we consider we develop Pyomo models to find optimal solutions to smaller but economically relevant problems. We use the problem of packing boxes onto shortest possible shelf of fixed width.

# ## Preamble: Install Pyomo and a solver
# 
# This cell selects and verifies a global SOLVER for the notebook.
# 
# If run on Google Colab, the cell installs Pyomo and HiGHS, then sets SOLVER to 
# use the Highs solver via the appsi module. If run elsewhere, it assumes Pyomo and CBC
# have been previously installed and sets SOLVER to use the CBC solver via the Pyomo 
# SolverFactory. It then verifies that SOLVER is available.

# In[1]:


import sys

if 'google.colab' in sys.modules:
    get_ipython().system('pip install pyomo >/dev/null 2>/dev/null')
    get_ipython().system('pip install highspy >/dev/null 2>/dev/null')

    from pyomo.contrib import appsi
    SOLVER = appsi.solvers.Highs(only_child_vars=False)
    
else:
    from pyomo.environ import SolverFactory
    SOLVER = SolverFactory('cbc')

assert SOLVER.available(), f"Solver {SOLVER} is not available."


# ## Problem Statment
# 
# Imagine a collection of $N$ boxes that are to placed on shelf. The shelf depth is $D$, and the dimensions of the boxes are $(w_i, d_i)$ for $i=0, \ldots, N-1$. The boxes can be rotated, if needed, to fit on the shelf. How wide of a shelf is needed?
# 
# We will start by creating a function to generate a table of $N$ boxes. For concreteness, we assume the dimensions are in millimeters.

# In[2]:


import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# random seed
random.seed(1842)

# generate boxes
def generate_boxes(N, max_width=200, max_depth=200):
    boxes = pd.DataFrame()
    boxes["w"] = [random.randint(0.2*max_width, max_width) for i in range(N)]
    boxes["d"] = [random.randint(0.2*max_depth, max_depth) for i in range(N)]
    return boxes

N = 8
boxes = generate_boxes(8)
display(boxes)

# set shelf width as a multiple of the deepest box
D = 2*boxes["d"].max()
print("Shelf Depth = ", D)


# ## A lower and upper bounds on shelf width
# 
# A lower bound on the shelf width is established by from the area required to place all boxes on the shelf.
# 
# $$W_{lb} = \frac{1}{D}\sum_{i=0}^{N-1} w_i d_i$$
# 
# An upper bound is established by aligning the boxes along the front of the shelf without rotation. To set the stage for later calculations, the position of the rectangle on the shelf is defined by bounding box $(x_{i,1}, y_{i,1})$ and $(x_{i,2}, y_{i,2})$ extending from the lower left corner to the upper right corner. 
# 
# $$
# \begin{align*}
# x_{i,2} & = x_{i,1} + w_i \\
# y_{i,2} & = y_{i,1} + d_i \\
# \end{align*}
# $$
# 
# An additional binary variable $r_i$ designates whether the rectangle has been rotated. The following cell performs these calculations to create and display a data frame showing the bounding boxes.

# In[3]:


def pack_boxes_V0(boxes):
    soln = boxes.copy()
    soln["x1"] = soln["w"].cumsum() - soln["w"]
    soln["x2"] = soln["w"].cumsum()
    soln["y1"] = 0
    soln["y2"] = soln["d"]
    soln["r"] = 0
    return soln

def show_boxes(soln, D):
    """Show bounding boxes on a diagram of the shelf."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    for i, x, y, w, h, r in zip(soln.index, soln["x1"], soln["y1"], soln["w"], soln["d"], soln["r"]):
        c = "g"
        if r:
            h, w = w, h
            c = "r"
        ax.add_patch(Rectangle((x, y), w, h, edgecolor="k", facecolor=c, alpha=0.6))
        xc = x + w/2
        yc = y + h/2
        ax.annotate(i, (xc, yc), color="w", weight="bold", fontsize=12, ha="center", va="center")
        
    W_lb = (soln["w"]*soln["d"]).sum()/D
    ax.set_xlim(0, 1.1*soln["w"].sum())
    ax.set_ylim(0, D*1.1)
    ax.axhline(D, label="shelf depth $D$", lw=0.8)
    ax.axvline(W_lb, label="lower bound $W_{lb}$", color="g", lw=0.8)
    ax.axvline(soln["x2"].max(), label="shelf width $W$", color="r", lw=0.8)
    ax.fill_between([0, ax.get_xlim()[1]], [D, D], color="b", alpha=0.1)
    ax.set_title(f"shelf width $W$ = {soln['x2'].max():.0f}")
    ax.set_xlabel("width")
    ax.set_ylabel("depth")
    ax.set_aspect('equal')
    ax.legend(loc="upper right")

soln = pack_boxes_V0(boxes)
display(soln)
show_boxes(soln, D)


# ## Modeling Strategy
# 
# At this point the reader may have some ideas on how to efficiently pack boxes on the shelf. For example, one might start by placing the larger boxes to left edge of the shelf, then rotating and placing the smaller boxes with a goal of minimized the occupied with of the shelf. 
# 
# Modeling for optimization takes a different approach. The strategy is to describe constraints that must be satisfied for any solution to the problem, then let the solver find the a choice for the decision variables that minimizes width. The constraints include:
# 
# * The bounding boxes must fit within the boundaries of the shelf, and to the left of vertical line drawn at $x = W$.
# * The boxes can be rotated 90 degrees.
# * The boxes must not overlap in either the $x$ or $y$ dimensions.

# ## Version 1: A Pyomo model to line up the boxes
# 
# For this first Pyomo model we look to reproduce a lineup of the boxes on the shelf. In the case the problem is to minimize $W$ where
# 
# $$
# \begin{align*}
# & \min W \\
# \text{subject to:}\qquad\qquad \\
# x_{i, 2} & = x_{i, 1} + w_i  & \forall i\\
# x_{i, 2} & \leq W  & \forall i\\
# x_{i, 1}, x_{i, 2} & \geq 0  & \forall i \\
# \\
# [x_{i, 2} \leq x_{j,1}] & \veebar [ x_{j, 2} \leq x_{i, 1}] & \forall i < j \\
# \\
# \end{align*}
# $$
# 
# This first model does not consider rotation or placement of the boxes in the $y$ dimension, so those decisions are not included. 
# 
# The disjunctive constraints specify relationships between $x_{i,1}$ and $x_{i,2}$ to prevent overlapping positions of boxes on the shelf. The disjunctions require that either that box $i$ is to the left of box $j$ or that box $j$ is the left of box $i$. This is specified as an exclusive or disjunction because both conditions can be true at the same time. This disjunctive relationship must hold for every pair of boxes that are different from each other, but specifying $i$ doesn't overlap with $j$ assures $j$ doesn't overlap $i$. So it is only necessary to specify disjunctions for all pairs $i, j$ where $i < j$.
# 
# The corresponding Pyomo model is a direct implementation of this model. One feature of the implementation is the use of a set `m.PAIRS` to identify the disjunctions. Defining this set simplifies coding for the corresponding disjunction.
# 

# In[4]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V1(boxes):
    
    # a simple upper bound on shelf width
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, bounds=(0, W_ub))

    @m.Objective()
    def minimize_width(m):
        return m.W
    
    @m.Constraint(m.BOXES)
    def bounding_box(m, i):
        return m.x2[i] == m.x1[i] + boxes.loc[i, "w"]
    
    @m.Constraint(m.BOXES)
    def width(m, i):
        return m.x2[i] <= m.W

    @m.Disjunction(m.PAIRS, xor=True)
    def no_overlap(m, i, j):
        return [m.x2[i] <= m.x1[j],
                m.x2[j] <= m.x1[i]]

    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    SOLVER.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [0 for i in boxes.index]
    soln["y2"] = soln["y1"] + soln["d"]
    soln["r"] = [0 for i in boxes.index]
    return soln

soln = pack_boxes_V1(boxes)
display(soln)
show_boxes(soln, D)


# ## Version 2: Rotating boxes
# 
# Rotating the boxes is an option for packing the boxes more tightly on the shelf. The boxes can be placed either in their original orientation or in a rotated orientation. This introduces a second exclusive or disjunction to the model that determines the orientation of the bounding box. A boolean indicator variable $r_i$ tracks which boxes were rotated which is used in the `show_boxes` function to show which boxes have been rotated.
# 
# $$
# \begin{align*}
# & \min W \\
# \text{subject to:}\qquad\qquad \\
# x_{i, 2} & \leq W  & \forall i\\
# x_{i, 1}, x_{i, 2} & \geq 0  & \forall i \\
# y_{i, 1} & = 0 & \forall i \\
# \\
# [x_{i, 2} \leq x_{j,1}] & \veebar [ x_{j, 2} \leq x_{i, 1}] & \forall i < j \\
# \\
# \begin{bmatrix}
# \neg r_i \\
# x_{i,2} = x_{i,1} + w_i\\
# y_{i,2} = y_{i,1} + d_i\\
# \end{bmatrix} & \veebar 
# \begin{bmatrix}
# r_i \\
# x_{i,2} = x_{i,1} + d_i\\
# y_{i,2} = y_{i,1} + w_i\\
# \end{bmatrix} & \forall i < j
# \end{align*}
# $$
# 
# For this version of the model the boxes will be lined up against the edge of the shelf with $y_{i,1} = 0$. Decision variables are now included in the model for rotation $r$ to the $y$ dimension of the bounding boxes.

# In[5]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V2(boxes):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.y1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.y2 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.r = pyo.Var(m.BOXES, domain=pyo.Boolean)

    @m.Objective()
    def minimize_width(m):
        return m.W
    
    @m.Constraint(m.BOXES)
    def width(m, i):
        return m.x2[i] <= m.W
    
    @m.Constraint(m.BOXES)
    def yloc(m, i):
        return m.y1[i] == 0
    
    @m.Disjunction(m.BOXES)
    def rotate(m, i):
        return [[m.r[i] == False, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "w"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "d"]],
                
                [m.r[i] == True, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "d"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "w"]]
               ]

    @m.Disjunction(m.PAIRS)
    def no_overlap(m, i, j):
        return [m.x2[i] <= m.x1[j],
                m.x2[j] <= m.x1[i]]

    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    SOLVER.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [m.y1[i]() for i in boxes.index]
    soln["y2"] = [m.y2[i]() for i in boxes.index]
    soln["r"] = [round(m.r[i]()) for i in boxes.index]
    
    return soln

soln = pack_boxes_V2(boxes)
display(soln)
show_boxes(soln, D)


# ## Version 3: Placing and Rotating boxes in two dimensions
# 
# Obviously the packages can be packed closer together by allowing boxes to be stacked deeper into the shelf. New constraints are needed to maintain the bounding boxes within the shelf depth, and to avoid overlaps in the $y$ dimension.
# 
# $$
# \begin{align*}
# & \min W \\
# \text{subject to:}\qquad\qquad \\
# x_{i, 2} & \leq W & \forall i\\
# y_{i, 2} & \leq D & \forall i\\
# x_{i, 1}, x_{i, 2} & \geq 0  & \forall i \\
# y_{i, 1}, y_{i, 2} & \geq 0 & \forall i \\
# \\
# \begin{bmatrix}
# x_{i, 2} \leq x_{j,1} \\
# \end{bmatrix}
# \veebar
# \begin{bmatrix}
# x_{j, 2} \leq x_{i, 1} \\
# \end{bmatrix}
# & \veebar 
# \begin{bmatrix}
# y_{i, 2} \leq y_{j, 1} \\
# \end{bmatrix}
# \veebar
# \begin{bmatrix}
# y_{j, 2} \leq y_{i, 1} \\
# \end{bmatrix}
# & \forall i < j \\
# \\
# \begin{bmatrix}
# \neg r_i \\
# x_{i,2} = x_{i,1} + w_i\\
# y_{i,2} = y_{i,1} + d_i\\
# \end{bmatrix} & \veebar 
# \begin{bmatrix}
# r_i \\
# x_{i,2} = x_{i,1} + d_i\\
# y_{i,2} = y_{i,1} + w_i\\
# \end{bmatrix} & \forall i < j
# \end{align*}
# $$
# 

# In[6]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V3(boxes, D):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()
    
    m.D = pyo.Param(mutable=True, initialize=D)

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.y1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.y2 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.r = pyo.Var(m.BOXES, domain=pyo.Boolean)

    @m.Objective()
    def minimize_width(m):
        return m.W
    
    @m.Constraint(m.BOXES)
    def width(m, i):
        return m.x2[i] <= m.W
    
    @m.Constraint(m.BOXES)
    def height(m, i):
        return m.y2[i] <= m.D
    
    @m.Disjunction(m.BOXES)
    def rotate(m, i):
        return [[m.r[i] == False, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "w"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "d"]],
                
                [m.r[i] == True, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "d"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "w"]]
               ]

    @m.Disjunction(m.PAIRS)
    def no_overlap(m, i, j):
        return [m.x2[i] <= m.x1[j],
                m.x2[j] <= m.x1[i],
                m.y2[i] <= m.y1[j],
                m.y2[j] <= m.y1[i]]

    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    SOLVER.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [m.y1[i]() for i in boxes.index]
    soln["y2"] = [m.y2[i]() for i in boxes.index]
    soln["r"] = [round(m.r[i]()) for i in boxes.index]
    return soln

soln = pack_boxes_V3(boxes, D)
display(soln)
show_boxes(soln, D)


# ## Advanced Topic: Symmetry Breaking
# 
# One of the issues in combinatorial problem is the challenge of symmetries. A symmetry is a situation where a change in solution configuration leaves the objective unchanged. Strip packing problems are especially susceptible to symmetries.
# 
# Symmetries can significantly increase the effort needed to find and and verify an optimal solution. Trespalacios and Grossmann recently presented modification to the strip packing problem to reduce the number of symmetries. This is described in the following paper and implemented in the accompanying Pyomo model.
# 
# Trespalacios, F., & Grossmann, I. E. (2017). Symmetry breaking for generalized disjunctive programming formulation of the strip packing problem. Annals of Operations Research, 258(2), 747-759.
# 

# In[7]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V4(boxes, D):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()
    
    m.D = pyo.Param(mutable=True, initialize=D)

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.y1 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.y2 = pyo.Var(m.BOXES, bounds=(0, W_ub))
    m.r = pyo.Var(m.BOXES, domain=pyo.Boolean)

    @m.Objective()
    def minimize_width(m):
        return m.W
    
    @m.Constraint(m.BOXES)
    def width(m, i):
        return m.x2[i] <= m.W
    
    @m.Constraint(m.BOXES)
    def height(m, i):
        return m.y2[i] <= m.D
    
    @m.Disjunction(m.BOXES)
    def rotate(m, i):
        return [[m.r[i] == False, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "w"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "d"]],
                
                [m.r[i] == True, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "d"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "w"]]
               ]

    @m.Disjunction(m.PAIRS)
    def no_overlap(m, i, j):
        return [[m.x2[i] <= m.x1[j]],
                
                [m.x2[j] <= m.x1[i]],
                
                [m.y2[i] <= m.y1[j],
                 m.x2[i] >= m.x1[j] + 1,
                 m.x2[j] >= m.x1[i] + 1],
                
                [m.y2[j] <= m.y1[i],
                 m.x2[i] >= m.x1[j] + 1,
                 m.x2[j] >= m.x1[i] + 1]
               ]

    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    SOLVER.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [m.y1[i]() for i in boxes.index]
    soln["y2"] = [m.y2[i]() for i in boxes.index]
    soln["r"] = [round(m.r[i]()) for i in boxes.index]
    return soln

soln = pack_boxes_V4(boxes, D)
display(soln)
show_boxes(soln, D)


# In[ ]:




