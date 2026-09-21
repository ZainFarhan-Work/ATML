"""Source-only ERM: cross-entropy on the three source domains, no target term."""


def alignment_loss(*_args, **_kwargs):
    """No alignment term; kept so every method shares one training loop."""
    return 0.0
