#!/usr/bin/env python
# coding: utf-8

# # McCormick Envelopes

# ## McCormick Envelope
# 
# Let $w = xy$ with upper and lower bounds on $x$
# 
# $$
# \begin{align*}
# x_1 \leq x \leq x_2 \\
# y_1 \leq y \leq y_2 \\
# \end{align*}
# $$
# 
# The "McCormick envelope" is a convex region satisfying the constraints
# 
# $$
# \begin{align*}
# w & \geq y_1 x + x_1 y - x_1 y_1 \\
# w & \geq y_2 x + x_2 y - x_2 y_2 \\
# w & \leq y_2 x + x_1 y - x_1 y_2 \\
# w & \leq y_1 x + x_2 y - x_2 y_1 \\
# \end{align*}
# $$
# 
# The following cells attempt to illustrate how this works.

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from matplotlib import style

n = 10

x1, x2 = 0.5, 10
y1, y2 = 0.5, 10
X, Y = np.meshgrid(np.linspace(x1, x2, n+1), np.linspace(y1, y2, n+1))

fig, ax = plt.subplots()
cp = ax.contourf(X, Y, X*Y, cmap=cm.cool, levels=n)
fig.colorbar(cp)

ax.axis('equal')
ax.set_xlim(0, x2 + x1)
ax.set_ylim(0, y2 + y1)
ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Bilinear function x*y')


# In[4]:


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from matplotlib import style

n = 10

x1, x2 = 0, 1
y1, y2 = 0, 1
X, Y = np.meshgrid(np.linspace(x1, x2, n+1), np.linspace(y1, y2, n+1))

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(10,10))

# surface plot
ax.plot_surface(X, Y, X*Y, alpha=1, cmap=cm.cool)
ax.plot_wireframe(X, Y, X*Y, lw=.3)

# annotate axis
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('w = x * y')
ax.view_init(elev=20, azim=-10)

# corner points (clockwise a -> b -> c -> d -> a)
a = np.array([x1, y1, x1*y1])
b = np.array([x1, y2, x1*y2])
c = np.array([x2, y2, x2*y2])
d = np.array([x2, y1, x2*x1])

def plot_line(a, b, color='r'):
    ax.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], lw=4, color=color, solid_capstyle="round")
    
# four edges
plot_line(a, b)
plot_line(b, c)
plot_line(c, d)
plot_line(d, a)

# catty corners
plot_line(b, d)
plot_line(a, c)

def show_surf(a, b, c):
    x = np.array([a[0], b[0], c[0]])
    y = np.array([a[1], b[1], c[1]])
    z = np.array([a[2], b[2], c[2]])
    ax.plot_trisurf(x, y, z, alpha=0.2)

show_surf(a, b, c)
show_surf(a, b, d)
show_surf(a, c, d)
show_surf(b, c, d)

plot_line([x1, y1, 0], a, 'k')
plot_line([x1, y2, 0], b, 'k')
plot_line([x2, y2, 0], c, 'k')
plot_line([x2, y1, 0], d, 'k')

#


# In[ ]:





# In[ ]:




