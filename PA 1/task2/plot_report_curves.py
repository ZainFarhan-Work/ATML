"""One compact training-curve figure for the report: Task 2 (source loss, alignment/domain loss) and
Task 3 (source loss, MMD penalty), from the saved per-epoch histories."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "Local Reports" / "PA1" / "figures" / "training_curves.png"


def load(task, name):
    rows = json.load(open(ROOT / task / "results" / f"history_{name}.json"))
    return [r["epoch"] for r in rows], rows


def series(task, name, key):
    epochs, rows = load(task, name)
    return epochs, [r[key] for r in rows]


T2 = [("source_only", "Source-only", "#7a7a7a"), ("dan", "DAN", "#eb6834"), ("dann", "DANN", "#1baf7a"), ("cdan", "CDAN", "#e8a000")]
T3 = [("dan_dg", r"DAN-DG (collapsed)", "#c0392b"), ("dan_dg_bwgrad", "DAN-DG (final)", "#2a78d6"), ("sam", "SAM", "#1baf7a")]

fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.75))
for name, label, color in T2:
    axes[0].plot(*series("task2", name, "cls_loss"), color=color, linewidth=1.3, marker="o", markersize=2, label=label)
    axes[1].plot(*series("task2", name, "align_loss"), color=color, linewidth=1.3, marker="o", markersize=2)
for name, label, color in T3:
    axes[2].plot(*series("task3", name, "cls_loss"), color=color, linewidth=1.3, marker="o", markersize=2, label=label)
    if name != "sam":
        axes[3].plot(*series("task3", name, "align_loss"), color=color, linewidth=1.3, marker="o", markersize=2)
for axis, title in zip(axes, ["Task 2: source classification loss", "Task 2: alignment / domain loss",
                              "Task 3: source classification loss", "Task 3: MMD penalty"]):
    axis.set_title(title, fontsize=6.3)
    axis.set_xlabel("epoch", fontsize=6)
    axis.tick_params(labelsize=5.5)
    axis.grid(alpha=0.25)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
axes[0].legend(fontsize=5.2, frameon=False)
axes[2].legend(fontsize=5.2, frameon=False)
fig.tight_layout(pad=0.4, w_pad=0.7)
fig.savefig(OUT, dpi=250, facecolor="white")
print(f"wrote {OUT}")
