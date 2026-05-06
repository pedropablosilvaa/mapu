import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath("src"))
from mapu import metaMDS, ordiplot, ordihull, ordiellipse, ordispider

print("Loading data...")
# Read dune data
df = pd.read_csv("../dune.csv")
# Generate fake environmental groups for the 20 sites
np.random.seed(42)
groups = np.random.choice(["Forest", "Grassland", "Wetland"], size=20)

print("Running NMDS...")
# Run NMDS
points = metaMDS(df, distance="bray", k=2, n_init=1)

print("Plotting...")
fig, ax = plt.subplots(figsize=(10, 8))

# 1. Base plot
ax = ordiplot(points, groups=groups, ax=ax, title="NMDS of Dune Dataset", s=100)

# 2. Add Hulls
ax = ordihull(points, groups=groups, ax=ax, alpha=0.1)

# 3. Add Ellipses
ax = ordiellipse(points, groups=groups, ax=ax, alpha=0.2, kind="sd")

# 4. Add Spider webs
ax = ordispider(points, groups=groups, ax=ax, linestyle="--", alpha=0.5)

plt.savefig("nmds_plot.png", dpi=300, bbox_inches="tight")
print("Plot saved to nmds_plot.png")
