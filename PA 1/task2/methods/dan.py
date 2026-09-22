"""DAN: multi-kernel MMD between source and target features."""

import torch


def median_bandwidth(distances, detach=True):
    """Median pairwise squared distance in the combined batch.

    detach=True (the default, used by every Task 2 run and by the manual-spec
    Task 3 run) treats the bandwidth as a constant within each step. With
    detach=False the gradient also flows through the bandwidth, which makes the
    loss exactly invariant to rescaling the features.
    """
    median = distances.median() if not detach else distances.detach().median()
    return median.clamp(min=1e-8)


def mmd_loss(
    source_features,
    target_features,
    multipliers=(0.5, 1.0, 2.0),
    detach_bandwidth=True,
):
    """Squared MMD with a sum of RBF kernels at 0.5/1/2 x the median bandwidth.

    The kernel trick evaluates similarities directly, so phi is never built.
    """
    combined = torch.cat([source_features, target_features], dim=0)
    distances = torch.cdist(combined, combined).pow(2)
    bandwidth = median_bandwidth(distances, detach=detach_bandwidth)

    kernel = sum(torch.exp(-distances / (multiplier * bandwidth)) for multiplier in multipliers)

    n = len(source_features)
    ss = kernel[:n, :n].mean()
    tt = kernel[n:, n:].mean()
    st = kernel[:n, n:].mean()
    return ss + tt - 2 * st
