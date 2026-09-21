"""Feature similarity analysis."""

import torch
import torch.nn.functional as F


def cosine_stability(clean_features, transformed_features):
    """Mean cosine similarity between paired clean and transformed representations.

    This is the manual's I_T: it asks whether the representation moved, which is a
    different question from whether the prediction changed.
    """
    return F.cosine_similarity(clean_features, transformed_features, dim=1).mean().item()


def random_pair_floor(features, seed=6304, samples=10000):
    """Mean cosine between unrelated images, as a reference point for I_T.

    Without it, a cosine of 0.6 cannot be read as high or low: feature spaces
    differ in how much of the sphere they use.
    """
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(len(features), (samples,), generator=generator)
    right = torch.randint(len(features), (samples,), generator=generator)
    keep = left != right
    return (
        F.cosine_similarity(features[left[keep]], features[right[keep]], dim=1)
        .mean()
        .item()
    )
