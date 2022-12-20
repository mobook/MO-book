import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

# describe the betting wheel
sectors = {
    "A": {"p": 1/2, "b": 3},
    "B": {"p": 1/3, "b": 2},
    "C": {"p": 1/6, "b": 6},
}

fig, ax = plt.subplots(1, 1)
a = np.pi/6
for s in sectors.keys():
    b = a + 2*np.pi*sectors[s]["p"]
    t = np.linspace(a, b, 40)
    c = plt.plot(np.cos(t), np.sin(t))[0].get_color()
    for k in range(1, len(t)):
        xpts = [0, np.cos(t[k-1]), np.cos(t[k])]
        ypts = [0, np.sin(t[k-1]), np.sin(t[k])]
        p = patches.Polygon(np.array([xpts, ypts]).T )
        p.set_color(c)
        ax.add_patch(p) 
    ax.text(0.55*np.cos((a + b)/2), 0.55*np.sin((a + b)/2), sectors[s]["b"], 
            fontsize=20, color="w", weight="bold", va="center", ha="center")
    p = patches.Polygon([[1.02, 0], [1.2, 0.2], [1.2, -0.2], [1.02, 0]])
    ax.plot(0, 0, 'w.', ms=100)
    ax.plot(0, 0, 'k.', ms=100, alpha=0.2)
    ax.plot(0, 0, 'k.', ms=50)
    p.set_color('k')
    ax.add_patch(p)
    a = b

ax.set_aspect(1)
ax.axis("off")

plt.savefig("investment-wheel.png", bbox_inches="tight", pad_inches=0)
