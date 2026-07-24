"""Tests for the wobble score, focused on the asymmetry (D-007)."""

import pytest

from wobblelab.wobble import model_wobble, wobble_score


def test_undecidable_gates_dispersion_off():
    # rain: high dispersion but undecidable -> dispersion must NOT count.
    r = wobble_score(dispersion=0.42, margin_max=0.72, answer=None)
    assert r["observational"] is None
    assert r["wobble"] == pytest.approx(1 - 0.72)  # interventional only
    assert r["driver"] == "interventional"


def test_decidable_dispersion_counts():
    # a decidable fact that's 50/50 on rerun but phrasing-stable -> max wobble.
    r = wobble_score(dispersion=0.5, margin_max=1.0, answer="yes")
    assert r["observational"] == pytest.approx(1.0)
    assert r["wobble"] == pytest.approx(1.0)
    assert r["driver"] == "observational"


def test_gating_flips_the_score():
    # identical measurement, only the label differs.
    dec = wobble_score(dispersion=0.15, margin_max=0.87, answer="yes")
    und = wobble_score(dispersion=0.15, margin_max=0.87, answer=None)
    assert dec["wobble"] == pytest.approx(0.30)  # dispersion drives it
    assert und["wobble"] == pytest.approx(0.13)  # only phrasing
    assert dec["wobble"] > und["wobble"]


def test_knife_edge_is_interventional():
    r = wobble_score(dispersion=0.10, margin_max=0.47, answer=None)
    assert r["wobble"] == pytest.approx(0.53)
    assert r["driver"] == "interventional"


def test_score_ci_takes_channel_max():
    r = wobble_score(
        dispersion=0.15,
        margin_max=0.69,
        answer="yes",
        disp_ci=(0.10, 0.22),
        margin_ci=(0.55, 0.82),
    )
    # interventional CI = (1-0.82, 1-0.55) = (0.18, 0.45); observational = (0.20, 0.44)
    assert r["wobble_ci"] == (pytest.approx(0.20), pytest.approx(0.45))


def test_model_wobble_aggregates_and_ranks():
    rows = [
        {"name": "a", "wobble": 0.53, "driver": "interventional"},
        {"name": "b", "wobble": 0.16, "driver": "interventional"},
        {"name": "c", "wobble": 0.30, "driver": "observational"},
    ]
    m = model_wobble(rows)
    assert m["max_wobble"] == 0.53
    assert m["worst_offenders"][0]["name"] == "a"
    assert m["n_prompts"] == 3


# --- wobble_factor: the report-card headline (worst-case combine) ---

from wobblelab import wobble_factor  # noqa: E402


def test_wobble_factor_is_worst_case():
    f = wobble_factor(reorder=0.365, position_swing=0.667, run_spread=0.198)
    assert f["factor"] == 0.667  # the worst signal, not an average
    assert f["driver"] == "position_swing"
    assert f["band"] == "high"
    assert f["components"]["reorder"] == 0.365  # decomposition retained


def test_wobble_factor_bands_and_clamping():
    assert wobble_factor(a=0.05)["band"] == "low"
    assert wobble_factor(a=0.3)["band"] == "moderate"
    assert wobble_factor(a=1.4)["factor"] == 1.0  # clamped into [0,1]
    assert wobble_factor()["factor"] == 0.0  # no signals -> zero, no crash
