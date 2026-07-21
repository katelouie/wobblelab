"""Tests for the report surface, driven by MockProvider -- no model, fully deterministic.

The logic must be provable with fake models: an "always A" model has to produce a maximal
position swing and an A-only chosen distribution; a "knows the answer" model has to score
100% with zero squish; and variable-option-count items (the TruthfulQA shape) must not crash
the fixed-width assumptions the harness used to make.
"""

import pytest

from squishlab import (
    MockProvider,
    MultipleChoiceTask,
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
    assert r.interventional_squish == pytest.approx(
        0.0
    )  # same content under every shuffle
    assert r.observational_squish == pytest.approx(0.0)  # deterministic
    assert r.position_swing == pytest.approx(0.0)  # right at every slot -> flat
    assert all(c == pytest.approx(0.25) for c in r.chosen_distribution)


def test_always_A_model_shows_position_bias():
    biased = MockProvider(always_position(0), name="always-A")
    r = evaluate(biased, ITEMS, scoring="ll", benchmark="toy")
    assert r.accuracy == pytest.approx(0.25)  # correct only when answer sits at A
    assert r.accuracy_by_position[0] == pytest.approx(1.0)
    assert all(a == pytest.approx(0.0) for a in r.accuracy_by_position[1:])
    assert r.position_swing == pytest.approx(1.0)  # maximal swing
    assert r.chosen_distribution[0] == pytest.approx(1.0)  # only ever picks A


def test_gen_and_ll_agree_for_a_deterministic_mock():
    biased = MockProvider(always_position(2), name="always-C")
    g = evaluate(biased, ITEMS, scoring="gen", n_rerun=3, benchmark="toy")
    ll = evaluate(biased, ITEMS, scoring="ll", benchmark="toy")
    assert g.accuracy == pytest.approx(ll.accuracy)
    assert g.n_rerun == 3 and ll.n_rerun == 1  # ll collapses reruns
    assert g.observational_squish == pytest.approx(
        0.0
    )  # mock is deterministic across seeds


def test_accuracy_ci_brackets_the_point_estimate():
    r = evaluate(MockProvider(always_position(0)), ITEMS, scoring="ll")
    lo, hi = r.accuracy_ci
    assert lo <= r.accuracy <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_unparseable_model_scores_zero_not_crash():
    silent = MockProvider(lambda prompt, seed: None, name="silent")
    r = evaluate(silent, ITEMS, scoring="ll")
    assert r.accuracy == pytest.approx(0.0)
    assert sum(r.chosen_distribution) == pytest.approx(0.0)


def test_variable_option_counts_do_not_crash():
    # The TruthfulQA shape: items with DIFFERENT numbers of options. The old fixed-width
    # harness assumed items[0]'s count for everyone; this must handle the mix.
    mixed = [
        MCItem(
            id="k3", question="three?", options=("CORRECT", "w", "w2"), answer_idx=0
        ),
        MCItem(
            id="k5",
            question="five?",
            options=("a", "b", "CORRECT", "d", "e"),
            answer_idx=2,
        ),
        MCItem(
            id="k4", question="four?", options=("a", "CORRECT", "c", "d"), answer_idx=1
        ),
    ]
    r = evaluate(MockProvider(picks_option_containing("CORRECT")), mixed, scoring="ll")
    assert r.accuracy == pytest.approx(
        1.0
    )  # perfect model still perfect regardless of width
    assert r.interventional_squish == pytest.approx(0.0)
    assert len(r.accuracy_by_position) == 5  # widened to the largest item
    # A slot that only the 5-option item reaches (E) is still counted, not dropped.
    assert r.accuracy_by_position[4] == pytest.approx(1.0)


def test_many_options_use_letters_past_H():
    # 12 options exercises the letters-past-H generalization (old cap was 8).
    opts = tuple("CORRECT" if i == 9 else f"w{i}" for i in range(12))
    item = MCItem(id="big", question="twelve?", options=opts, answer_idx=9)
    r = evaluate(MockProvider(picks_option_containing("CORRECT")), [item], scoring="ll")
    assert r.accuracy == pytest.approx(1.0)
    assert len(r.accuracy_by_position) == 12


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
    d = reports[0].to_dict()
    assert d["model"] == "perfect" and 0.0 <= d["accuracy"] <= 1.0


def test_explicit_task_argument_is_honored():
    r = evaluate(
        MockProvider(picks_option_containing("CORRECT")),
        ITEMS,
        task=MultipleChoiceTask(scoring="ll"),
        benchmark="toy",
    )
    assert r.task == "mc:ll" and r.accuracy == pytest.approx(1.0)


def test_evaluate_rejects_bad_scoring_and_empty():
    with pytest.raises(ValueError):
        evaluate(MockProvider(always_position(0)), ITEMS, scoring="bogus")
    with pytest.raises(ValueError):
        evaluate(MockProvider(always_position(0)), [], scoring="ll")
