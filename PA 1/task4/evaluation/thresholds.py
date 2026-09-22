"""Rejection threshold calibrated on CIFAR-10 validation data only.

tau = 95th percentile of unknownness on the known validation set; accept when
u(x) <= tau. By construction this aims to accept 95% of known examples.
"""

import numpy as np

PERCENTILE = 95.0


def threshold(validation_scores, percentile=PERCENTILE):
    return float(np.percentile(np.asarray(validation_scores), percentile))


def acceptance_rate(scores, tau):
    """Fraction accepted as known, in percent."""
    return float((np.asarray(scores) <= tau).mean()) * 100
