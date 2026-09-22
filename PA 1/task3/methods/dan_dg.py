"""DAN-DG: MMD between every pair of observed source domains. Sketch is never seen."""

import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task2.methods.dan import mmd_loss  # same discrepancy and kernels as Task 2


def pairwise_mmd(domain_features, detach_bandwidth=True):
    """Mean MMD^2 over the three unordered source pairs.

    Bandwidths are computed per pair from that pair's own median distance, as in
    Task 2, so only the information available changes - not the measure.
    """
    pairs = list(combinations(domain_features, 2))
    total = sum(
        mmd_loss(first, second, detach_bandwidth=detach_bandwidth)
        for first, second in pairs
    )
    return total / len(pairs)


def lambda_at(config, progress):
    """lambda_DG at training progress p in [0, 1].

    "constant" (default) is the manual's specification. "ramp" reuses the DANN
    schedule 2 / (1 + exp(-10 p)) - 1, so alignment pressure arrives after the
    classifier has formed features: 0.17 x lambda_DG after epoch 1, 0.68 at epoch 5,
    0.93 at epoch 10, 0.97 at epoch 13 and 1.0 by epoch 30.
    """
    weight = config["lambda_dg"]
    if config.get("lambda_schedule", "constant") == "ramp":
        weight *= 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return weight
