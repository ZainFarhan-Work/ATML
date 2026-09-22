"""AUROC for known vs unknown, and closed-set accuracy."""

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(known_scores, unknown_scores):
    """Probability that a random unknown scores more unknown than a random known.

    Larger unknownness must mean more novel, so unknowns are the positive class.
    """
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    s = np.concatenate([np.asarray(known_scores), np.asarray(unknown_scores)])
    return roc_auc_score(y, s) * 100


def closed_set_accuracy(logits, labels):
    """CSA over the ten known-class logits only."""
    return float((logits.argmax(dim=1) == labels).float().mean()) * 100
