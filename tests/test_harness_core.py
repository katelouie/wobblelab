"""Tests for the new Harness core: rendering, extraction, and running against mock providers.

GPU-free. The mocks keep the logic honest: an 'always slot 1' policy must show position
dependence; a 'picks the right letter' policy must score 100%.
"""

import pytest

from wobblelab.benchmark import LETTERS, MCItem
from wobblelab.catalog import GPQA_AUTHORS, GPQA_DIAMOND, Benchmark
from wobblelab.engine import accuracy, coverage, run_harness
from wobblelab.harness import (
    Harness,
    OrderSpec,
    PromptSpec,
    ScoringSpec,
    extract,
    render_prompt,
)
from wobblelab.provider import MockProvider, always_position

ITEMS = [
    MCItem(
        id="q0",
        question="Q0?",
        options=("Paris", "Rome", "Berlin", "Madrid"),
        answer_idx=0,
    ),
    MCItem(id="q1", question="Q1?", options=("a", "b", "c", "d"), answer_idx=1),
    MCItem(id="q2", question="Q2?", options=("w", "x", "y", "z"), answer_idx=2),
]

# A minimal gen harness whose extraction matches a bare letter -- so MockProvider (which returns
# a bare "A"/"B"/...) parses cleanly.
SIMPLE = Harness(
    name="simple",
    prompt=PromptSpec(choices_header="\n", answer_trigger="Answer with a letter."),
    scoring=ScoringSpec(method="gen", extraction_patterns=(r"\b([A-Za-z])\b",)),
)
LL = Harness(
    name="ll",
    prompt=PromptSpec(choices_header="\n"),
    scoring=ScoringSpec(method="ll_letter"),
)


class FormatMock:
    """Emits 'The correct answer is (X)' so the GPQA cascade has a parenthesized letter to parse."""

    def __init__(self, policy):
        self._policy = policy

    def ask(self, prompt, seed):
        pos = self._policy(prompt, seed)
        return (
            ""
            if pos is None
            else f"Reasoning... The correct answer is ({LETTERS[pos]})."
        )

    def rank_letters(self, prompt, n_options, seed=0, top_logprobs=20):
        pos = self._policy(prompt, seed)
        return (pos, {LETTERS[pos]: 0.0}) if pos is not None else (None, {})

    def config(self):
        return {"provider": "format-mock"}


def test_render_gpqa_authors_prompt():
    p = render_prompt(GPQA_AUTHORS, ITEMS[0])
    assert p.startswith("What is the correct answer to this question: Q0?")
    assert "\n\nChoices:\n(A) Paris\n(B) Rome\n(C) Berlin\n(D) Madrid" in p
    assert p.endswith(
        'Format your response as follows: "The correct answer is (insert answer here)"'
    )


def test_extract_gpqa_cascade():
    assert extract(GPQA_AUTHORS, "The correct answer is (B).", 4) == 1
    assert extract(GPQA_AUTHORS, "...blah... the answer is (D)", 4) == 3
    assert (
        extract(GPQA_AUTHORS, "I think it's Paris", 4) is None
    )  # no parenthesized letter
    assert extract(GPQA_AUTHORS, "The correct answer is (E)", 4) is None  # out of range


def test_run_harness_gen_shows_position_dependence():
    outs = run_harness(MockProvider(always_position(1)), SIMPLE, ITEMS)
    assert [o.parsed_pos for o in outs] == [1, 1, 1]  # always slot B
    assert [o.correct for o in outs] == [
        False,
        True,
        False,
    ]  # right only when answer is at B
    assert accuracy(outs) == pytest.approx(1 / 3)
    assert coverage(outs) == 1.0  # every response parsed


def test_run_harness_ll_letter_path():
    outs = run_harness(MockProvider(always_position(2)), LL, ITEMS)
    assert [o.parsed_pos for o in outs] == [2, 2, 2]
    assert [o.raw for o in outs] == [None, None, None]  # ll has no generated text
    assert accuracy(outs) == pytest.approx(1 / 3)  # right only for q2 (answer at C)


def test_gpqa_gen_end_to_end_with_format_mock():
    # a "perfect" mock: always picks the correct answer's canonical slot
    perfect = FormatMock(
        lambda prompt, seed: 0
    )  # canonical order -> correct is at its own slot
    outs = run_harness(perfect, GPQA_AUTHORS, [ITEMS[0]])  # q0 answer at slot 0
    assert outs[0].parsed_pos == 0 and outs[0].correct and coverage(outs) == 1.0


def test_ll_choice_not_implemented():
    h = Harness(name="x", prompt=PromptSpec(), scoring=ScoringSpec(method="ll_choice"))
    with pytest.raises(NotImplementedError):
        run_harness(MockProvider(always_position(0)), h, ITEMS[:1])


def test_concurrency_is_deterministic():
    prov = MockProvider(always_position(1))
    seq = run_harness(prov, SIMPLE, ITEMS, concurrency=1)
    par = run_harness(prov, SIMPLE, ITEMS, concurrency=4)
    assert seq == par  # concurrency changes speed, never the outcomes


def test_permutation_order_maps_back_to_originals():
    h = Harness(
        name="perm",
        prompt=SIMPLE.prompt,
        scoring=SIMPLE.scoring,
        order=OrderSpec(
            kind="permutation", order=(2, 0, 1, 3)
        ),  # slot A shows option[2]
    )
    outs = run_harness(
        MockProvider(always_position(0)), h, ITEMS
    )  # model always picks slot A
    assert all(o.parsed_pos == 0 for o in outs)
    assert all(o.parsed_orig == 2 for o in outs)  # slot A held original option 2
    # correct only when the dataset's answer_idx is 2 (q2)
    assert [o.correct for o in outs] == [False, False, True]


def test_benchmark_wiring():
    assert GPQA_DIAMOND.name == "gpqa_diamond"
    assert GPQA_DIAMOND.canonical is GPQA_AUTHORS
    assert GPQA_DIAMOND.landmarks["authors"] is GPQA_AUTHORS
    assert isinstance(GPQA_DIAMOND, Benchmark)
