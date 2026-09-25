"""ROC figure drawn at its printed size (5.5 in wide) so labels stay readable in the report; same data as plot_osr_figures.roc."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from evaluation.thresholds import threshold  # noqa: E402
from plot_osr_figures import KNOWN, NEAR, FAR, MUTED, SECONDARY, load  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "Local Reports" / "PA1" / "figures" / "task4_roc_print.png"

scores = load()
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05))
for axis, (name, s) in zip(axes, scores.items()):
    tau = threshold(s["val"])
    for color, label, unknown in ((NEAR, "near", s["near"]), (FAR, "far", s["far"])):
        y = np.r_[np.zeros(len(s["test"])), np.ones(len(unknown))]
        fpr, tpr, _ = roc_curve(y, np.r_[s["test"], unknown])
        axis.plot(fpr, tpr, color=color, linewidth=1.6, label=label)
        axis.plot((s["test"] > tau).mean(), (unknown > tau).mean(), "o", color=color, markeredgecolor="white", markersize=5)
    axis.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=0.9)
    axis.set_title(name, fontsize=8, color=SECONDARY)
    axis.set_xlabel("known rejected (FPR)", fontsize=7, color=SECONDARY)
    axis.set_aspect("equal")
    axis.tick_params(labelsize=6.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
axes[0].set_ylabel("unknowns rejected (TPR)", fontsize=7, color=SECONDARY)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(pad=0.5, w_pad=0.6)
fig.savefig(OUT, dpi=250, facecolor="white")
print("wrote", OUT)
