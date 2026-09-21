"""SAM: non-adaptive sharpness-aware minimisation (Foret et al., 2021)."""

import torch


@torch.no_grad()
def ascend(parameters, rho):
    """Move to theta + eps, the worst point within an L2 ball of radius rho."""
    norm = torch.norm(
        torch.stack([p.grad.norm(p=2) for p in parameters if p.grad is not None]), p=2
    )
    scale = rho / (norm + 1e-12)
    perturbations = []
    for parameter in parameters:
        if parameter.grad is None:
            perturbations.append(None)
            continue
        step = parameter.grad * scale
        parameter.add_(step)
        perturbations.append(step)
    return perturbations, norm


@torch.no_grad()
def descend(parameters, perturbations):
    """Restore theta before the optimiser applies the perturbed-point gradient."""
    for parameter, step in zip(parameters, perturbations):
        if step is not None:
            parameter.sub_(step)
