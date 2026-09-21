"""DAN: multi-kernel MMD between source and target features."""

import torch


def median_bandwidth(distances):
    """Median pairwise squared distance in the combined batch."""
    return distances.detach().median().clamp(min=1e-8)


def mmd_loss(source_features, target_features, multipliers=(0.5, 1.0, 2.0)):
    """Squared MMD with a sum of RBF kernels at 0.5/1/2 x the median bandwidth.

    The kernel trick evaluates similarities directly, so phi is never built.
    """
    combined = torch.cat([source_features, target_features], dim=0)
    distances = torch.cdist(combined, combined).pow(2)
    bandwidth = median_bandwidth(distances)

    kernel = sum(torch.exp(-distances / (multiplier * bandwidth)) for multiplier in multipliers)

    n = len(source_features)
    ss = kernel[:n, :n].mean()
    tt = kernel[n:, n:].mean()
    st = kernel[:n, n:].mean()
    return ss + tt - 2 * st
