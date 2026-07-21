"""Tests for the report surface, driven by MockProvider -- no model, fully deterministic.

The logic must be provable with fake models: an "always A" model has to produce a maximal
position swing and an A-only chosen distribution; a "knows the answer" model has to score
100% with zero reorder squish. If those hold, evaluate() is measuring what it claims to.
"""

import pytest

from squishlab import (
    MockProvider,
    always_position,
    compare,
    compare_markdown,
    evaluate,
    picks_option_containing,
)
from squishlab.benchmark import MCItem

# Correct option is marked "CORRECT" so picks_option_containing can play a perfect model;
# distractors are distinct and never contain the needle.
ITEMS = [
    MCItem(
        id=f"q{i}",
        question=f"Question {i}?",
        options=("CORRECT", "w1", "w2", "w3"),
        answer_idx=0,
    )
    for i in range(8)
]


def test_perfect_model_scores_100_with_no_squish():
    perfect = MockProvider(picks_option_containing("CORRECT"), name="perfect")
    r = evaluate(perfect, ITEMS, scoring="ll", benchmark="toy")
    assert r.accuracy == pytest.approx(1.0)
    assert r.reorder_squish == pytest.approx(0.0)  # same content under every shuffle
    assert r.position_swing == pytest.approx(0.0)  # right at every slot -> flat
    # It follows the answer wherever it moves, so every slot gets chosen equally.
    assert all(c == pytest.approx(0.25) for c in r.chosen_distribution)


def test_always_A_model_shows_position_bias():
    biased = MockProvider(always_position(0), name="always-A")
    r = evaluate(biased, ITEMS, scoring="ll", benchmark="toy")
    # Correct only when the answer happens to sit at slot A: 1 of 4 positions.
    assert r.accuracy == pytest.approx(0.25)
    assert r.accuracy_by_position[0] == pytest.approx(
        1.0
    )  # perfect when answer is at A
    assert all(a == pytest.approx(0.0) for a in r.accuracy_by_position[1:])
    assert r.position_swing == pytest.approx(1.0)  # maximal swing
    assert r.chosen_distribution[0] == pytest.approx(1.0)  # only ever picks A


def test_generation_path_runs_and_agrees_with_ll_for_a_deterministic_mock():
    # A MockProvider is deterministic, so gen and ll must give the same accuracy.
    biased = MockProvider(always_position(2), name="always-C")
    g = evaluate(biased, ITEMS, scoring="gen", n_rerun=3, benchmark="toy")
    ll = evaluate(biased, ITEMS, scoring="ll", benchmark="toy")
    assert g.accuracy == pytest.approx(ll.accuracy)
    assert g.n_rerun == 3 and ll.n_rerun == 1  # ll collapses reruns


def test_accuracy_ci_brackets_the_point_estimate():
    biased = MockProvider(always_position(0))
    r = evaluate(biased, ITEMS, scoring="ll")
    lo, hi = r.accuracy_ci
    assert lo <= r.accuracy <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_unparseable_model_scores_zero_not_crash():
    # A policy returning None (nothing parseable) must not divide-by-zero or throw.
    silent = MockProvider(lambda prompt, seed: None, name="silent")
    r = evaluate(silent, ITEMS, scoring="ll")
    assert r.accuracy == pytest.approx(0.0)
    assert sum(r.chosen_distribution) == pytest.approx(0.0)


def test_compare_and_markdown():
    reports = compare(
        {
            "perfect": MockProvider(picks_option_containing("CORRECT")),
            "always-A": MockProvider(always_position(0)),
        },
        ITEMS,
        scoring="ll",
        benchmark="toy",
    )
    assert [r.model for r in reports] == ["perfect", "always-A"]
    md = compare_markdown(reports)
    assert "perfect" in md and "always-A" in md and "accuracy" in md
    # round-trip through to_dict stays JSON-shaped
    d = reports[0].to_dict()
    assert d["model"] == "perfect" and 0.0 <= d["accuracy"] <= 1.0


def test_evaluate_rejects_bad_scoring_and_empty():
    with pytest.raises(ValueError):
        evaluate(MockProvider(always_position(0)), ITEMS, scoring="bogus")
    with pytest.raises(ValueError):
        evaluate(MockProvider(always_position(0)), [], scoring="ll")
