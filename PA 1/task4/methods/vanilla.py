"""Vanilla closed-set baseline: plain cross-entropy on the ten known classes."""

import torch.nn.functional as F


def loss(logits, labels):
    return F.cross_entropy(logits, labels)
