"""Energy: unknownness = -logsumexp over the logits (uses all of them)."""

import torch


def score(logits, features=None, stats=None):
    return (-torch.logsumexp(logits, dim=1)).cpu()
