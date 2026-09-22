"""Common local sharpness proxy, identical for every Task 3 model.

One normalised gradient-ascent step of radius 0.05 on a fixed validation batch:
    delta = L(theta + eps) - L(theta),  eps = 0.05 * grad / ||grad||.
This is a standardised local diagnostic, not proof that one model is globally flatter.
"""

import torch
import torch.nn.functional as F

RADIUS = 0.05
PER_DOMAIN = 32
SEED = 6304


def _loss(features, classifier, images, labels):
    return F.cross_entropy(classifier(features(images)), labels)


def sharpness(features, classifier, images, labels, radius=RADIUS):
    """Increase in cross-entropy after one normalised ascent step."""
    features.eval()
    classifier.eval()
    parameters = [p for p in list(features.parameters()) + list(classifier.parameters())
                  if p.requires_grad]
    for parameter in parameters:
        parameter.grad = None

    base = _loss(features, classifier, images, labels)
    base.backward()

    with torch.no_grad():
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in parameters if p.grad is not None]), p=2
        )
        scale = radius / (norm + 1e-12)
        steps = []
        for parameter in parameters:
            step = parameter.grad * scale if parameter.grad is not None else None
            if step is not None:
                parameter.add_(step)
            steps.append(step)

        perturbed = _loss(features, classifier, images, labels)

        for parameter, step in zip(parameters, steps):
            if step is not None:
                parameter.sub_(step)

    for parameter in parameters:
        parameter.grad = None
    return float(perturbed - base.detach()), float(base.detach()), float(norm)
