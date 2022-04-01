#!/usr/bin/env python
# coding: utf-8

# # Economic Order Quantity

# Kuo, Y. J., & Mittelmann, H. D. (2004). Interior point methods for second-order cone programming and OR applications. Computational Optimization and Applications, 28(3), 255-285. https://link.springer.com/content/pdf/10.1023/B:COAP.0000033964.95511.23.pdf
# 
# Letchford, A. N., & Parkes, A. J. (2018). A guide to conic optimisation and its applications. RAIRO-Operations Research, 52(4-5), 1087-1106. http://www.cs.nott.ac.uk/~pszajp/pubs/conic-guide.pdf
# 
# Ziegler, H. (1982). Solving certain singly constrained convex optimization problems in production planning. Operations Research Letters, 1(6), 246-252.
# 
# Bretthauer, K. M., & Shetty, B. (1995). The nonlinear resource allocation problem. Operations research, 43(4), 670-683. https://www.jstor.org/stable/171693?seq=1
# 
# 

# ## Model
# 
# History
# 
# * Formulated in 1982 by Ziegler. Extension of classic EOQ problem
# * Reformulated in 2004 SOCP by Kuo & Mittleman, cited by Letchford
# * See Bretthauer for related applications
# 
# Demonstrates
# 
# * formulation hyperbolic SOCP
# * Familiar to any business major
# * Significant business application
# 
# $$
# \begin{align*}
# \min & \quad \sum_{i=1}^n \frac{h x_i}{2} + \frac{c_i d_i}{x_i} \\
# \\
# \text{s.t.} & \quad \sum_{i=1}^n b_i x_i  \leq b_0 \\
# & l_i \leq x_i \leq u_i & \forall i\in 1, \dots, n
# \\
# \end{align*}
# $$
# 

# In[ ]:




