"""MLS: unknownness = -max logit (keeps absolute magnitude, unlike MSP)."""


def score(logits, features=None, stats=None):
    return (-logits.max(dim=1).values).cpu()
