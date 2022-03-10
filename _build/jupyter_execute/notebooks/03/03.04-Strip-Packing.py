#!/usr/bin/env python
# coding: utf-8

# # Strip Packing: Placing Boxes on a Shelf
# 
# *Strip packing* (SP) refers to the problem of packing rectanges onto a two dimensional strip of fixed width. 
# 
# Many variants of this problem have been studied, the most basic is the pack a set of rectangles onto the shortest possible strip without rotation. Other variants allow rotation of the rectangles, require a packing that allows cutting the rectanges out of the strip with edge-to-edge cuts (guillotine packing), extends the strip to three dimensions, or extends the packing to non-rectangular shapes.
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
# Finding optimial solutions to strip packing problems is combinatorially difficult. Strip packing belongs to a class of problems called "NP-hard" for which known solution algorithms require effort that grows exponentially with problem size. For that reason much research on strip packing has been directed towards practical heurestic algorithms for finding good, though not optimal, for industrial applications.
# 
# Here we consider we develop Pyomo models to find optimal solutions to smaller but economically relevant problems. We use the problem of packing boxes onto shortest possible shelf of fixed width.

# ## Problem Statment
# 
# Imagine a collection of $N$ boxes that are to placed on shelf. The shelf depth is $D$, and the dimensions of the boxes are $(w_i, d_i)$ for $i=0, \ldots, N-1$. The boxes can be rotated, if needed, to fit on the shelf. How wide of a shelf is needed?
# 
# 
# We'll start by creating a function to generate a table of $N$ boxes. For concreteness, we assume the dimensions are in millimeters.

# In[122]:


import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# generate boxes
def generate_boxes(N, max_width=200, max_depth=200):
    boxes = pd.DataFrame()
    boxes["w"] = [random.randint(0.2*max_width, max_width) for i in range(N)]
    boxes["d"] = [random.randint(0.2*max_depth, max_depth) for i in range(N)]
    return boxes

N = 8
boxes = generate_boxes(8)
display(boxes)

D = 2*boxes["d"].max()
print("Shelf Depth = ", D)


# ## Version 0: Line up the boxes on the shelf
# 
# The first solution is simply to line up the boxes on the shelf. For comparison purposes, a lower bound on the needed width is easy to compute.
# 
# $$W_{lb} = \frac{1}{D}\sum_{i=0}^{N-1} w_i d_i$$
# 
# The position of the rectange on the shelf is defined by bounding box $(x_{i,1}, y_{i,1})$ and $(x_{i,2}, y_{i,2})$ extending from the lower left corncer to the upper right corner. 
# 
# $$
# \begin{align*}
# x_{i,2} & = x_{i,1} + w_i \\
# y_{i,2} & = y_{i,2} + d_i \\
# \end{align*}
# $$
# 
# An additional binary variable $r_i$ designates whether the rectangle has been rotated.

# In[123]:


def pack_boxes_V0(boxes):
    soln = boxes.copy()
    soln["x1"] = soln["w"].cumsum() - soln["w"]
    soln["x2"] = soln["x1"] + soln["w"]
    soln["y1"] = 0
    soln["y2"] = soln["y1"] + soln["d"]
    soln["r"] = 0
    return soln

def show_boxes(boxes, D):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for i, x, y, w, h, r in zip(boxes.index, boxes["x1"], boxes["y1"], 
                                boxes["w"], boxes["d"], boxes["r"]):
        c = "g"
        if r == 1:
            h, w = w, h
            c = "r"
        ax.add_patch(Rectangle((x, y), w, h, edgecolor="k", facecolor=c, alpha=0.6))
        ax.annotate(i, (x + w/2, y + h/2), color="w", weight="bold", 
                    fontsize=12, ha="center", va="center")
        
    W_lb = (boxes["w"]*boxes["d"]).sum()/D
    ax.set_xlim(0, boxes["w"].sum())
    ax.set_ylim(0, D*1.1)
    ax.axhline(D, label="$D$")
    ax.axvline(W_lb, label="$W_{lb}$")
    ax.fill_between([W_lb, ax.get_xlim()[1]], [D, D], color="b", alpha=0.1)
    ax.fill_between([0, W_lb], [D, D], color="y", alpha=0.1)
    ax.set_xlabel("width")
    ax.set_ylabel("depth")
    ax.set_aspect('equal')
    ax.legend(loc="upper right")

soln = pack_boxes_V0(boxes)
display(soln)
show_boxes(soln, D)


# ## Version 1: A Pyomo model to line up the boxes
# 
# For this first Pyomo model, we simply look to reproduce a lineup of the boxes on the shelf. In the case the problem is minimize $W$ where
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
# The disjuctive constraints prevent overlapping positions of boxes on the shelf.
# 
# The corresponding Pyomo model is a direct implementation of this model. One feature of the implementation is the use of a set `m.PAIRS` to identify the disjunctions. Defining this set simplifies coding the corresponding disjunction.
# 

# In[124]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V1(boxes):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))

    @m.Objective()
    def minimize_width(m):
        return m.W
    
    @m.Constraint(m.BOXES)
    def bounding_box(m, i):
        return m.x2[i] == m.x1[i] + boxes.loc[i, "w"]
    
    @m.Constraint(m.BOXES)
    def width(m, i):
        return m.x2[i] <= m.W

    @m.Disjunction(m.PAIRS)
    def no_overlap(m, i, j):
        return [m.x2[i] <= m.x1[j],
                m.x2[j] <= m.x1[i]]

    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    solver = pyo.SolverFactory("gurobi_direct")
    solver.solve(m)

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
# Rotating the boxes is an allowed action in this shelf packing application. This introduces an additional disjuction to determine the orientation of the bounding box. We include a binary indicator variable $r_i$ that will be used to track which boxes were rotated.
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
# r_i = 0 \\
# x_{i,2} = x_{i,1} + w_i\\
# y_{i,2} = y_{i,1} + d_i\\
# \end{bmatrix} & \veebar 
# \begin{bmatrix}
# r_i = 1 \\
# x_{i,2} = x_{i,1} + d_i\\
# y_{i,2} = y_{i,1} + w_i\\
# \end{bmatrix} & \forall i < j
# \end{align*}
# $$
# 
# The Pyomo model implementation now adds the decision variables for rotation and $y$ position.

# In[125]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V2(boxes):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.y1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.y2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.r = pyo.Var(m.BOXES, domain=pyo.Binary)

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
        return [[m.r[i] == 0, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "w"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "d"]],
                
                [m.r[i] == 1, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "d"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "w"]]
               ]

    @m.Disjunction(m.PAIRS)
    def no_overlap(m, i, j):
        return [m.x2[i] <= m.x1[j],
                m.x2[j] <= m.x1[i]]

    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    solver = pyo.SolverFactory("gurobi_direct")
    solver.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [m.y1[i]() for i in boxes.index]
    soln["y2"] = [m.y2[i]() for i in boxes.index]
    soln["r"] = [m.r[i]() for i in boxes.index]
    
    return soln

soln = pack_boxes_V2(boxes)
display(soln)
show_boxes(soln, D)


# ## Version 3: Placing and Rotating boxes in two dimensions
# 
# Obiously the packages can be packed closer together by allowing boxes to be stacked deeper into the shelf. New constraints are needed to maintain the bounding boxes within the shelf depth, and to avoid overlaps in the $y$ dimension.
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
# r_i = 0 \\
# x_{i,2} = x_{i,1} + w_i\\
# y_{i,2} = y_{i,1} + d_i\\
# \end{bmatrix} & \veebar 
# \begin{bmatrix}
# r_i = 1 \\
# x_{i,2} = x_{i,1} + d_i\\
# y_{i,2} = y_{i,1} + w_i\\
# \end{bmatrix} & \forall i < j
# \end{align*}
# $$
# 

# In[144]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V3(boxes, D):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()
    
    m.D = pyo.Param(mutable=True, initialize=D)

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.y1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.y2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.r = pyo.Var(m.BOXES, domain=pyo.Binary)

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
        return [[m.r[i] == 0, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "w"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "d"]],
                
                [m.r[i] == 1, 
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
    solver = pyo.SolverFactory("gurobi_direct")
    solver.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [m.y1[i]() for i in boxes.index]
    soln["y2"] = [m.y2[i]() for i in boxes.index]
    soln["r"] = [m.r[i]() for i in boxes.index]
    return soln

soln = pack_boxes_V3(boxes, D)
display(soln)
show_boxes(soln, D)


# ## Advanced Topic: Symmetry Breaking
# 
# One of the issues in combinatorial problem is the challenge of symmetries. A symmetry is a sitatuion where a change in solution configuration leaves the objective unchanged. Strip packing problems are especially susceptable to symmetries.
# 
# Symmetries can significantly increase the effort needed to find and and verify an optimal solution. Trespalacios and Grossmann recently presented modification to the strip packing problem to reduce the number of symmetries. This is described in the following paper and implemented in the accompanying Pyomo model.
# 
# While symmetry breaking does speed up solutioin of sample problems, the netw
# 
# Trespalacios, F., & Grossmann, I. E. (2017). Symmetry breaking for generalized disjunctive programming formulation of the strip packing problem. Annals of Operations Research, 258(2), 747-759.
# 

# In[153]:


import pyomo.environ as pyo
import pyomo.gdp as gdp

def pack_boxes_V4(boxes, D):
    
    W_ub = boxes["w"].sum()

    m = pyo.ConcreteModel()
    
    m.D = pyo.Param(mutable=True, initialize=D)

    m.BOXES = pyo.Set(initialize=boxes.index)
    m.PAIRS = pyo.Set(initialize=m.BOXES * m.BOXES, filter=lambda m, i, j: i < j)

    m.W = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.x2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.y1 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.y2 = pyo.Var(m.BOXES, domain=pyo.NonNegativeReals, bounds=(0, W_ub))
    m.r = pyo.Var(m.BOXES, domain=pyo.Binary)

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
        return [[m.r[i] == 0, 
                 m.x2[i] == m.x1[i] + boxes.loc[i, "w"],
                 m.y2[i] == m.y1[i] + boxes.loc[i, "d"]],
                
                [m.r[i] == 1, 
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
    solver = pyo.SolverFactory("gurobi_direct")
    solver.solve(m)

    soln = boxes.copy()
    soln["x1"] = [m.x1[i]() for i in boxes.index]
    soln["x2"] = [m.x2[i]() for i in boxes.index]
    soln["y1"] = [m.y1[i]() for i in boxes.index]
    soln["y2"] = [m.y2[i]() for i in boxes.index]
    soln["r"] = [m.r[i]() for i in boxes.index]
    return soln

soln = pack_boxes_V4(boxes, D)
display(soln)
show_boxes(soln, D)

