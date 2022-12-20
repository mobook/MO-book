import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

alpha = 0.4

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.set_aspect(1)
ax.axis("off")

ax.add_patch(patches.Circle((0, 0), 0.12, color="k", alpha=alpha))
ax.add_patch(patches.Circle((0.5, 0.25), 0.12, color="k", alpha=alpha))
ax.add_patch(patches.Circle((0.5, -0.25), 0.12, color="k", alpha=alpha))

ax.text(0, 0, "1", fontsize=20, color="w", weight="bold", va="center", ha="center")
ax.text(0.5, 0.25, "b", fontsize=20, color="w", weight="bold", va="center", ha="center")
ax.text(0.5, -0.25, "0", fontsize=20, color="w", weight="bold", va="center", ha="center")

ax.arrow(0.14, 0.07, 0.18, 0.09, lw=10, alpha=alpha)
ax.arrow(0.14, -0.07, 0.18, -0.09, lw=10, alpha=alpha)

ax.text(0.2, 0.2, "p", ha="center", va="center", fontsize=20)
ax.text(0.2, -0.2, "1-p", ha="center", va="center", fontsize=20)

ax.text(-0.2, 0, "wager = 1", ha="right", va="center", fontsize=20)
ax.text(0.7, 0.25, "payout = b", ha="left", va="center", fontsize=20)
ax.text(0.7, -0.25, "payout = 0", ha="left", va="center", fontsize=20)

ax.set_xlim(-.2, 1.0)
ax.set_ylim(-0.45, 0.45)

plt.savefig("kelly-criterion.png", bbox_inches="tight", pad_inches=0)
