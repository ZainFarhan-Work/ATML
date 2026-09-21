"""DANN: marginal adversarial alignment via a gradient-reversal discriminator."""

import torch
import torch.nn.functional as F

from models.domain_discriminator import reverse_gradient


def domain_loss(discriminator, source_features, target_features, alpha):
    """Binary domain classification on reversed features; both domains contribute."""
    features = torch.cat([source_features, target_features], dim=0)
    labels = torch.cat(
        [
            torch.zeros(len(source_features), dtype=torch.long),
            torch.ones(len(target_features), dtype=torch.long),
        ]
    ).to(features.device)
    logits = discriminator(reverse_gradient(features, alpha))
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return F.cross_entropy(logits, labels), accuracy
