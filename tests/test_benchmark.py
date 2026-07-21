"""Tests for the option-permutation machinery — the round-trip must be exact."""

from squishlab.benchmark import (
    MCItem,
    format_prompt,
    modal,
    parse_answer,
    presented_to_original,
)

ITEM = MCItem(
    id="t1",
    question="Capital of France?",
    options=("Paris", "Rome", "Berlin", "Madrid"),
    answer_idx=0,
)


def test_format_letters_track_presentation_position():
    p = format_prompt(ITEM, (2, 0, 3, 1))  # Berlin, Paris, Madrid, Rome
    assert "A. Berlin" in p and "B. Paris" in p and "C. Madrid" in p and "D. Rome" in p
    assert "Answer with only the letter (A/B/C/D)." in p


def test_parse_answer_extracts_letter():
    assert parse_answer("B", 4) == 1
    assert parse_answer("The answer is D.", 4) == 3
    assert parse_answer("(a)", 4) == 0
    assert parse_answer("I'm not sure", 4) is None
    assert parse_answer("E", 4) is None  # out of range for 4 options


def test_permutation_round_trip_recovers_correct_answer():
    # Under this order Paris (original idx 0, the answer) sits at presentation position 1 -> "B".
    order = (2, 0, 3, 1)
    chosen_pos = parse_answer("B", 4)
    assert presented_to_original(chosen_pos, order) == ITEM.answer_idx
    # A robust model picking the same *content* under the identity order:
    assert presented_to_original(parse_answer("A", 4), (0, 1, 2, 3)) == 0


def test_modal():
    val, frac = modal([0, 0, 1, 0, None])
    assert val == 0 and frac == 0.75  # 3 of 4 decided
    assert modal([None, None]) == (None, 0.0)
