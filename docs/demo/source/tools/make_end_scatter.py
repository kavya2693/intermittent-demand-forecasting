"""Draw the faded end-card background: every SKU on the ADI / CV2 plane, from outputs/sku_segments.csv.

Decoration only (no axes).
Usage: python3 tools/make_end_scatter.py <repo>/outputs/sku_segments.csv assets/scatter-end.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"smooth": "#3fd8a8", "erratic": "#7fa8ff", "intermittent": "#b9a2ff", "lumpy": "#ff8a73"}
s = pd.read_csv(sys.argv[1])
fig = plt.figure(figsize=(5.8, 4.85), dpi=200)
ax = fig.add_axes((0, 0, 1, 1))
for seg, g in s.groupby("segment"):
    ax.scatter(np.log(g["adi"]), g["cv2"].clip(upper=3.2), s=9, c=COL[seg], alpha=0.75, linewidths=0)
ax.axvline(np.log(1.32), color="#eef2f6", lw=1.2, ls=(0, (4, 4)), alpha=0.6)
ax.axhline(0.49, color="#eef2f6", lw=1.2, ls=(0, (4, 4)), alpha=0.6)
ax.set_xlim(-0.1, np.log(16))
ax.set_ylim(-0.1, 3.3)
ax.axis("off")
fig.savefig(sys.argv[2], transparent=True)
print("wrote", sys.argv[2])
