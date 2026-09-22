"""MSP: unknownness = 1 - max softmax probability."""

import torch.nn.functional as F


def score(logits, features=None, stats=None):
    return (1.0 - F.softmax(logits, dim=1).max(dim=1).values).cpu()
