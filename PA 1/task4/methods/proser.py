"""PROSER: classifier placeholders + manifold-mixup data placeholders.

Following Zhou et al. (2021). The dummy classifiers occupy one extra logit
column, taken as the strongest dummy response, so the known-class logits stay
untouched and CSA can still be computed from them alone.
"""

import torch
import torch.nn.functional as F

BETA = 1.0   # classifier-placeholder weight
GAMMA = 0.1  # data-placeholder weight
DUMMY_INDEX = -1  # the appended dummy column


def with_dummy(known_logits, dummy_logit):
    """[10 known logits | strongest dummy response]."""
    return torch.cat([known_logits, dummy_logit], dim=1)


def classifier_placeholder_loss(known_logits, dummy_logit, labels):
    """Keep the true class on top; make a dummy the runner-up once it is removed.

    Term 1 is ordinary cross-entropy over the 11 columns. Term 2 masks the true
    class to -inf and asks the dummy column to win what remains, which is what
    teaches the placeholders to sit just outside each known class.
    """
    combined = with_dummy(known_logits, dummy_logit)
    correct = F.cross_entropy(combined, labels)

    masked = combined.clone()
    masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    dummy_target = torch.full_like(labels, combined.shape[1] - 1)
    placeholder = F.cross_entropy(masked, dummy_target)
    return correct + BETA * placeholder


def data_placeholder_loss(mixed_known_logits, mixed_dummy_logit):
    """Mixed between-class features should be claimed by the dummy classifier."""
    combined = with_dummy(mixed_known_logits, mixed_dummy_logit)
    target = torch.full(
        (len(combined),), combined.shape[1] - 1, dtype=torch.long, device=combined.device
    )
    return GAMMA * F.cross_entropy(combined, target)


def detection_score(known_logits, dummy_logit):
    """Placeholder-based unknownness: dummy evidence minus best known evidence.

    Larger means more unknown, matching the convention used by the other scores.
    """
    return (dummy_logit.squeeze(1) - known_logits.max(dim=1).values).cpu()
