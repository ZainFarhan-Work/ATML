"""Domain discriminator and gradient-reversal layer shared by DANN and CDAN."""

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    """Identity forwards; scales and flips the gradient backwards."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def reverse_gradient(x, alpha):
    return GradientReversal.apply(x, alpha)


def schedule(progress, maximum=1.0):
    """alpha(p) = 2 / (1 + exp(-10p)) - 1, scaled by the maximum strength."""
    import math

    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def build_discriminator(input_dim, hidden=256, dropout=0.5):
    """256-unit hidden layer, ReLU, dropout 0.5, two-class output."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 2),
    )
