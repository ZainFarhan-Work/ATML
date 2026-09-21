"""Representation analysis."""

import numpy as np
from sklearn.manifold import TSNE

SEED = 6304
PERPLEXITY = 30
# Cosine metric: CLIP embeddings are L2-normalized, so angle is the meaningful
# distance, and using one metric keeps the three backbones comparable in method
# (not in coordinates - those are never compared across separate fits).
METRIC = "cosine"


def fit_projection(features, seed=SEED, perplexity=PERPLEXITY):
    """One 2-D t-SNE fitted to clean and transformed features together."""
    projector = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=METRIC,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return projector.fit_transform(np.asarray(features, dtype=np.float32))


def plot_panels(points, condition, labels, classes, title, path):
    """One panel per transform: clean circles vs transformed crosses, colored by class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 10 classes exceeds the 8-slot categorical palette, so use a CVD-oriented
    # 10-colour set; marker style carries the condition as secondary encoding.
    colors = sns.color_palette("colorblind", len(classes))
    transformed = [name for name in dict.fromkeys(condition) if name != "clean"]

    figure, axes = plt.subplots(1, len(transformed), figsize=(4 * len(transformed), 4.2))
    axes = np.atleast_1d(axes)
    clean = condition == "clean"

    for axis, name in zip(axes, transformed):
        axis.scatter(
            points[clean, 0], points[clean, 1],
            c=[colors[i] for i in labels[clean]],
            s=9, marker="o", alpha=0.45, linewidths=0,
        )
        mask = condition == name
        axis.scatter(
            points[mask, 0], points[mask, 1],
            c=[colors[i] for i in labels[mask]],
            s=13, marker="x", alpha=0.85, linewidths=0.9,
        )
        axis.set_title(f"clean (o) vs {name} (x)", fontsize=9, color="#52514e")
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color("#898781")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[i], label=name)
        for i, name in enumerate(classes)
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=len(classes),
        frameon=False, fontsize=7, labelcolor="#52514e",
    )
    figure.suptitle(title, fontsize=10, color="#0b0b0b")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(figure)
    print(f"wrote {path}")
