"""DAN-DG: MMD between every pair of observed source domains. Sketch is never seen."""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task2.methods.dan import mmd_loss  # same discrepancy and kernels as Task 2


def pairwise_mmd(domain_features):
    """Mean MMD^2 over the three unordered source pairs.

    Bandwidths are computed per pair from that pair's own median distance, as in
    Task 2, so only the information available changes - not the measure.
    """
    pairs = list(combinations(domain_features, 2))
    total = sum(mmd_loss(first, second) for first, second in pairs)
    return total / len(pairs)
