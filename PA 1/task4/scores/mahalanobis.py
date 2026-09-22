"""Mahalanobis: distance to the nearest known-class feature cluster.

Class means and one shared diagonal covariance are estimated from *unaugmented*
CIFAR-10 training features, with 1e-6 added to every diagonal entry.
"""

import torch

EPSILON = 1e-6


def fit(features, labels, num_classes=10, epsilon=EPSILON):
    """Class means and the shared diagonal covariance, from training features."""
    means = torch.stack([features[labels == c].mean(dim=0) for c in range(num_classes)])
    centred = torch.cat([features[labels == c] - means[c] for c in range(num_classes)])
    variance = centred.var(dim=0, unbiased=False) + epsilon
    return {"means": means, "variance": variance}


def score(logits, features=None, stats=None):
    """min_c (f - mu_c)^T Sigma^-1 (f - mu_c) with diagonal Sigma."""
    difference = features.unsqueeze(1) - stats["means"].unsqueeze(0).to(features.device)
    weighted = difference.pow(2) / stats["variance"].to(features.device)
    return weighted.sum(dim=2).min(dim=1).values.cpu()
