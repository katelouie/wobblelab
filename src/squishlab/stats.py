"""Confidence-interval primitives for squishlab.

Small, dependency-light, and tested, because a framework that measures reliability
should not itself report point estimates without error bars. See lab-journal D-005.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

Z95 = 1.959963984540054  # standard-normal quantile for a 95% two-sided interval


def wilson_ci(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi).

    Preferred over the normal approximation for small n and proportions near 0/1,
    which is exactly the regime we live in (yes-rates near 0, 0.5, or 1).
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = Z95
) -> tuple[float, float]:
    """Newcombe's score interval for the difference of two proportions p1 - p2.

    Composes two Wilson intervals (Newcombe 1998, "method 10" / MOVER-Wilson).
    Robust with small samples and near-boundary proportions, where the naive
    Wald interval for a difference misbehaves. Returns (lo, hi) for p1 - p2.
    """
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def confident_shift(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> float:
    """Magnitude of the p1 - p2 shift we are statistically confident exists.

    0 if the difference CI straddles 0 (could be pure noise); otherwise the CI
    endpoint nearest 0 (the smallest shift consistent with the data). This is the
    noise-robust worst-case ingredient: it will not fire on sampling jitter.
    """
    lo, hi = newcombe_diff_ci(k1, n1, k2, n2, z)
    if lo > 0:
        return lo
    if hi < 0:
        return -hi
    return 0.0


def bootstrap_ci(
    data: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary statistic of a sample.

    Resamples ``data`` WITH REPLACEMENT ``n_boot`` times and recomputes ``statistic``.
    NOTE: this cannot invent statistical power you did not sample -- for a plain
    proportion it merely reproduces ``wilson_ci``. Its real use is *derived/composite*
    statistics with no clean closed form (the squish score, the model-level headline
    over prompts, benchmark accuracy across orderings), where you resample the recorded
    outcomes at zero additional model cost. See lab-journal D-008.
    """
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = sorted(
        statistic([data[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    )
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)
