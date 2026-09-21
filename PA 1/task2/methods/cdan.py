"""CDAN: adversarial alignment conditioned on the classifier's predictions."""

import torch
import torch.nn.functional as F

from models.domain_discriminator import reverse_gradient


def conditional_map(features, probabilities):
    """g(x) = vec(f (x) p): the multilinear map of feature and class posterior.

    Neither f nor p is detached, as the manual requires, so class structure
    receives adversarial gradient too.
    """
    outer = torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1))
    return outer.flatten(start_dim=1)


def domain_loss(discriminator, source_pair, target_pair, alpha):
    """Domain classification on the conditioned representation."""
    source = conditional_map(*source_pair)
    target = conditional_map(*target_pair)
    mapped = torch.cat([source, target], dim=0)
    labels = torch.cat(
        [
            torch.zeros(len(source), dtype=torch.long),
            torch.ones(len(target), dtype=torch.long),
        ]
    ).to(mapped.device)
    logits = discriminator(reverse_gradient(mapped, alpha))
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return F.cross_entropy(logits, labels), accuracy
