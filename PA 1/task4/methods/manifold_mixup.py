"""Manifold mixup between different-class examples, for PROSER's data placeholders."""

import numpy as np
import torch

ALPHA = 2.0  # Beta(2, 2), as the manual specifies


def different_class_pairs(labels, generator=None):
    """A permutation that pairs each example with one of a different class."""
    order = torch.randperm(len(labels), generator=generator)
    for _ in range(10):  # resample the offenders a few times, then drop them
        clash = labels[order] == labels
        if not clash.any():
            break
        order[clash] = order[clash][torch.randperm(int(clash.sum()), generator=generator)]
    return order, labels[order] != labels


def mix(hidden, order, rng=None):
    """h = lam * h_i + (1 - lam) * h_j with lam ~ Beta(2, 2)."""
    rng = rng or np.random
    lam = float(rng.beta(ALPHA, ALPHA))
    return lam * hidden + (1.0 - lam) * hidden[order], lam
