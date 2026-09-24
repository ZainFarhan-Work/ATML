"""Rescaled versions of the required OSR figure, from saved outputs only.

The original task4_score_distributions.png is left untouched. Its MSP panel is nearly
unreadable because 1 - max softmax probability is ~0 for most images, so almost all
the mass falls in the first bin on a linear axis. This writes two alternatives:
a log-y distribution figure with the validation threshold marked, and ROC curves.
"""

import sys
from pathlib import Path

TASK4 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK4))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from evaluation.thresholds import threshold  # noqa: E402
from scores import mahalanobis, mls, msp  # noqa: E402

CACHE_DIR = TASK4 / "cache"
FIGURE_DIR = TASK4 / "results" / "figures"
SHOWN = {"MSP": msp, "MLS": mls, "Mahalanobis": mahalanobis}

# Same series colours as every other figure: blue / orange / green.
KNOWN, NEAR, FAR = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, SECONDARY = "#898781", "#52514e"


def load():
    packed = torch.load(CACHE_DIR / "outputs_vanilla.pt")
    stats = mahalanobis.fit(packed["train"]["features"], packed["train"]["labels"])
    return {
        name: {s: mod.score(packed[s]["logits"], packed[s]["features"], stats).numpy()
               for s in ("val", "test", "near", "far")}
        for name, mod in SHOWN.items()
    }


def style(axis):
    axis.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)


def distributions(scores, bins=45):
    """Log-y density histograms with the validation-calibrated threshold marked."""
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for axis, (name, s) in zip(axes, scores.items()):
        pooled = np.concatenate([s["test"], s["near"], s["far"]])
        edges = np.linspace(pooled.min(), pooled.max(), bins + 1)
        for color, label, values in ((KNOWN, "known (test)", s["test"]),
                                     (NEAR, "near", s["near"]), (FAR, "far", s["far"])):
            density, _ = np.histogram(values, bins=edges, density=True)
            centres = (edges[:-1] + edges[1:]) / 2
            keep = density > 0  # zero-count bins cannot be drawn on a log axis
            axis.plot(centres[keep], density[keep], color=color, linewidth=1.8,
                      marker="o", markersize=2.5, label=label)
        tau = threshold(s["val"])
        axis.axvline(tau, color=SECONDARY, linestyle="--", linewidth=1.1)
        axis.text(tau, 0.97, r" $\tau$", transform=axis.get_xaxis_transform(),
                  color=SECONDARY, fontsize=9, va="top")
        axis.set_yscale("log")
        axis.set_title(name, fontsize=10, color=SECONDARY)
        axis.set_xlabel("unknownness", fontsize=9, color=MUTED)
        style(axis)
    axes[0].set_ylabel("density (log scale)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="upper right")
    figure.tight_layout()
    path = FIGURE_DIR / "task4_score_distributions_logy.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def roc(scores):
    """ROC per score (unknown = positive), with the 95%-TPR operating point marked."""
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for axis, (name, s) in zip(axes, scores.items()):
        tau = threshold(s["val"])
        for color, label, unknown in ((NEAR, "near", s["near"]), (FAR, "far", s["far"])):
            y = np.r_[np.zeros(len(s["test"])), np.ones(len(unknown))]
            fpr, tpr, _ = roc_curve(y, np.r_[s["test"], unknown])
            axis.plot(fpr, tpr, color=color, linewidth=2, label=label)
            # Operating point at the validation threshold: reject when u(x) > tau.
            axis.plot((s["test"] > tau).mean(), (unknown > tau).mean(), "o",
                      color=color, markeredgecolor="#fcfcfb", markersize=7)
        axis.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
        axis.set_title(name, fontsize=10, color=SECONDARY)
        axis.set_xlabel("known images rejected (FPR)", fontsize=9, color=MUTED)
        axis.set_aspect("equal")
        style(axis)
    axes[0].set_ylabel("unknowns rejected (TPR)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="lower right")
    figure.tight_layout()
    path = FIGURE_DIR / "task4_roc.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scores = load()
    distributions(scores)
    roc(scores)


if __name__ == "__main__":
    main()
