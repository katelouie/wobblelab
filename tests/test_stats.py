"""Sanity tests for the CI primitives. A reliability tool tests its own statistics."""

import pytest

from squishlab.stats import (
    bootstrap_ci,
    confident_shift,
    newcombe_diff_ci,
    wilson_ci,
)


def test_bootstrap_reproduces_wilson_for_a_proportion():
    # For a plain mean, the bootstrap CI should land near the Wilson interval --
    # confirming it invents no power, just estimates the same uncertainty.
    data = [1] * 70 + [0] * 30
    lo, hi = bootstrap_ci(data, lambda s: sum(s) / len(s), n_boot=4000, seed=1)
    wlo, whi = wilson_ci(70, 100)
    assert abs(lo - wlo) < 0.03 and abs(hi - whi) < 0.03


def test_bootstrap_handles_a_composite_statistic():
    # max of two channel-means: a stat with no clean closed-form CI.
    data = [(0.5, 0.1), (0.4, 0.2), (0.6, 0.05), (0.3, 0.3)]
    stat = lambda s: max(sum(a for a, _ in s) / len(s), sum(b for _, b in s) / len(s))  # noqa: E731
    lo, hi = bootstrap_ci(data, stat, n_boot=2000, seed=2)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_symmetric_midpoint():
    lo, hi = wilson_ci(50, 100)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-9)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_boundaries_stay_in_unit_interval():
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.4
    lo1, hi1 = wilson_ci(10, 10)
    assert hi1 == pytest.approx(1.0) and 0.6 < lo1 < 1.0


def test_wilson_empty():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_newcombe_clear_difference_excludes_zero():
    lo, hi = newcombe_diff_ci(9, 10, 1, 10)  # 0.9 vs 0.1
    assert lo > 0.0  # confident there IS a positive shift
    assert hi <= 1.0


def test_newcombe_no_difference_straddles_zero():
    lo, hi = newcombe_diff_ci(5, 10, 5, 10)  # identical proportions
    assert lo < 0.0 < hi


def test_confident_shift_fires_only_when_significant():
    assert confident_shift(9, 10, 1, 10) > 0.0  # clear shift
    assert confident_shift(5, 10, 5, 10) == 0.0  # noise, no confident shift
    # small samples with a modest gap: not confident -> 0
    assert confident_shift(6, 10, 4, 10) == 0.0
