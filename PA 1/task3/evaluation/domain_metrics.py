"""Accuracy and macro-F1 per source domain, plus mean and worst."""

import numpy as np

from shared.pacs_protocol import evaluate  # noqa: F401


def summarise(per_domain):
    """Mean and worst-domain accuracy / macro-F1 across the source validation sets."""
    accuracies = [r["accuracy"] for r in per_domain.values()]
    f1s = [r["macro_f1"] for r in per_domain.values()]
    return {
        "mean_source_acc": float(np.mean(accuracies)),
        "mean_source_f1": float(np.mean(f1s)),
        "worst_source_acc": float(np.min(accuracies)),
        "worst_source_f1": float(np.min(f1s)),
    }
