"""GCSC: the vanilla recipe with RandAugment(2, 9) only.

Nothing changes in the objective; the augmentation flag lives in the config and
is applied by data/cifar10.py, so this module exists to make that explicit.
"""

from methods.vanilla import loss  # noqa: F401
