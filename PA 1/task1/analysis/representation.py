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

    # 2x2 rather than 1xN: at NeurIPS text width a four-across strip leaves each
    # panel ~1.4 inches wide, which is unreadable in print.
    rows = int(np.ceil(len(transformed) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(9.5, 4.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    clean = condition == "clean"

    for axis, name in zip(axes, transformed):
        axis.scatter(
            points[clean, 0], points[clean, 1],
            c=[colors[i] for i in labels[clean]],
            s=26, marker="o", alpha=0.40, linewidths=0,
        )
        mask = condition == name
        axis.scatter(
            points[mask, 0], points[mask, 1],
            c=[colors[i] for i in labels[mask]],
            s=34, marker="x", alpha=0.9, linewidths=1.5,
        )
        axis.set_title(f"clean (o) vs {name} (x)", fontsize=14, color="#52514e")
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color("#898781")

    for axis in axes[len(transformed):]:
        axis.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   color=colors[i], label=name)
        for i, name in enumerate(classes)
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=5,
        frameon=False, fontsize=12, labelcolor="#52514e",
    )
    figure.suptitle(title, fontsize=15, color="#0b0b0b")
    figure.tight_layout(rect=(0, 0.10, 1, 0.97))
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(figure)
    print(f"wrote {path}")
