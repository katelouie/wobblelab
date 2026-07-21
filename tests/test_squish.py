"""Tests for the squish score, focused on the asymmetry (D-007)."""

import pytest

from squishlab.squish import model_squish, squish_score


def test_undecidable_gates_dispersion_off():
    # rain: high dispersion but undecidable -> dispersion must NOT count.
    r = squish_score(dispersion=0.42, margin_max=0.72, answer=None)
    assert r["observational"] is None
    assert r["squish"] == pytest.approx(1 - 0.72)  # interventional only
    assert r["driver"] == "interventional"


def test_decidable_dispersion_counts():
    # a decidable fact that's 50/50 on rerun but phrasing-stable -> max squish.
    r = squish_score(dispersion=0.5, margin_max=1.0, answer="yes")
    assert r["observational"] == pytest.approx(1.0)
    assert r["squish"] == pytest.approx(1.0)
    assert r["driver"] == "observational"


def test_gating_flips_the_score():
    # identical measurement, only the label differs.
    dec = squish_score(dispersion=0.15, margin_max=0.87, answer="yes")
    und = squish_score(dispersion=0.15, margin_max=0.87, answer=None)
    assert dec["squish"] == pytest.approx(0.30)  # dispersion drives it
    assert und["squish"] == pytest.approx(0.13)  # only phrasing
    assert dec["squish"] > und["squish"]


def test_knife_edge_is_interventional():
    r = squish_score(dispersion=0.10, margin_max=0.47, answer=None)
    assert r["squish"] == pytest.approx(0.53)
    assert r["driver"] == "interventional"


def test_score_ci_takes_channel_max():
    r = squish_score(
        dispersion=0.15,
        margin_max=0.69,
        answer="yes",
        disp_ci=(0.10, 0.22),
        margin_ci=(0.55, 0.82),
    )
    # interventional CI = (1-0.82, 1-0.55) = (0.18, 0.45); observational = (0.20, 0.44)
    assert r["squish_ci"] == (pytest.approx(0.20), pytest.approx(0.45))


def test_model_squish_aggregates_and_ranks():
    rows = [
        {"name": "a", "squish": 0.53, "driver": "interventional"},
        {"name": "b", "squish": 0.16, "driver": "interventional"},
        {"name": "c", "squish": 0.30, "driver": "observational"},
    ]
    m = model_squish(rows)
    assert m["max_squish"] == 0.53
    assert m["worst_offenders"][0]["name"] == "a"
    assert m["n_prompts"] == 3
