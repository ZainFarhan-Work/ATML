from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 6304, alpha: float = 0.05):
    """Mean and percentile CI, resampling the leading axis (questions / pairs)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot: int = 1000, seed: int = 6304, alpha: float = 0.05):
    """CI for mean(a - b) with the same resampled units for both arms."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)


def bootstrap_matrix_ci(per_unit: dict[str, np.ndarray], n_boot: int = 4000, seed: int = 6304, alpha: float = 0.05):
    """Resample units once and apply the same indices to every named arm.

    per_unit maps arm name -> array of shape [n_units]. Returns {arm: (mean, lo, hi)} plus
    resampled means so that callers can form differences between arms consistently.
    """
    arms = list(per_unit)
    n = len(per_unit[arms[0]])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = {k: np.asarray(per_unit[k], dtype=float)[idx].mean(axis=1) for k in arms}
    out = {
        k: (float(np.mean(per_unit[k])), float(np.quantile(boots[k], alpha / 2)), float(np.quantile(boots[k], 1 - alpha / 2)))
        for k in arms
    }
    return out, boots


def diff_ci_from_boots(boots_a: np.ndarray, boots_b: np.ndarray, mean_a: float, mean_b: float, alpha: float = 0.05):
    d = boots_a - boots_b
    return float(mean_a - mean_b), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
