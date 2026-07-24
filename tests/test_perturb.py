"""Tests for the perturbation strategies -- the interventional axis, per kind."""

import pytest

from squishlab import (
    FormalityShift,
    LexicalSwap,
    NaturalOrder,
    ParaphraseWithModel,
    Perturbation,
    RephraseInstruction,
    ReorderOptions,
    TranslateWithModel,
)
from squishlab.benchmark import MCItem
from squishlab.perturb import render

ITEM = MCItem(
    id="t",
    question="Capital of France?",
    options=("Paris", "Rome", "Berlin"),
    answer_idx=0,
)


class TextPerturber:
    """A perturber whose ask() returns text (MockProvider returns letters, no good here)."""

    def __init__(self, transform):
        self.transform = transform

    def ask(self, prompt, seed):
        return self.transform(prompt, seed)

    def rank_letters(self, *a, **k):
        return None, {}

    def config(self):
        return {"provider": "text-perturber"}


def test_reorder_is_position_probing_and_permutes():
    pres = ReorderOptions().present(ITEM)
    assert len(pres) == 3  # one per option slot
    assert all(p.probes_position for p in pres)
    assert all(p.kind == "reorder" for p in pres)
    for i, p in enumerate(pres):
        assert p.origin[i] == ITEM.answer_idx  # answer placed at slot i
        assert sorted(p.origin) == [0, 1, 2]  # a real permutation
        assert p.question == ITEM.question  # question untouched


def test_natural_order_is_one_identity_presentation_not_position_probing():
    pres = NaturalOrder().present(ITEM)
    assert len(pres) == 1
    p = pres[0]
    assert p.kind == "natural"
    assert p.origin == (0, 1, 2)  # identity: answer stays at its dataset slot
    assert not p.probes_position  # leaderboard number, adds no position-bias samples
    assert p.question == ITEM.question and p.options == ITEM.options


def test_rephrase_instruction_varies_only_the_ask():
    strat = RephraseInstruction()
    pres = strat.present(ITEM)
    assert len(pres) == len(strat.instructions)
    assert all(not p.probes_position for p in pres)  # doesn't move the answer
    assert all(p.origin == (0, 1, 2) for p in pres)  # identity order
    assert all(p.question == ITEM.question for p in pres)  # question untouched
    assert len({p.instruction for p in pres}) == len(pres)  # each instruction distinct
    assert isinstance(strat, Perturbation)


def test_render_letters_the_presented_options():
    p = ReorderOptions().present(ITEM)[1]  # answer (Paris) at slot B
    text = render(p)
    assert "B. Paris" in text and "Capital of France?" in text
    assert "Answer with only the letter (A/B/C)." in text


def test_paraphrase_uses_the_perturber_for_the_question():
    perturber = TextPerturber(
        lambda prompt, seed: f"Which city is France's capital? [v{seed}]"
    )
    pres = ParaphraseWithModel(perturber, n=2).present(ITEM)
    assert len(pres) == 2
    assert all(p.kind == "paraphrase" for p in pres)
    assert all("France's capital" in p.question for p in pres)  # question was reworded
    assert pres[0].options == ITEM.options  # options untouched
    assert pres[0].origin == (0, 1, 2)


def test_translate_transforms_question_and_options():
    perturber = TextPerturber(lambda prompt, seed: "<t>" + prompt.split("\n\n")[-1])
    pres = TranslateWithModel(perturber, "Latin").present(ITEM)
    assert len(pres) == 1
    assert pres[0].kind == "translate:Latin"
    assert pres[0].question.startswith("<t>")
    assert all(o.startswith("<t>") for o in pres[0].options)  # options translated too
    assert pres[0].origin == (0, 1, 2)


def test_empty_perturber_output_falls_back_to_original():
    pres = ParaphraseWithModel(TextPerturber(lambda p, s: "   "), n=1).present(ITEM)
    assert pres[0].question == ITEM.question  # blank rewrite -> keep the original


def test_paraphrase_rejects_nothing_but_is_a_perturbation():
    assert isinstance(
        ParaphraseWithModel(TextPerturber(lambda p, s: "x")), Perturbation
    )
    with pytest.raises(TypeError):
        ReorderOptions().present()  # needs an item


def test_formality_and_lexical_are_distinct_directed_rewrites():
    # Echo the directive so we can confirm each strategy sends a different instruction.
    def echo_directive(prompt, seed):
        return prompt.split("\n\n")[0]  # the directive line, before the question

    p = TextPerturber(echo_directive)
    formal = FormalityShift(p, n=1).present(ITEM)[0]
    lexical = LexicalSwap(p, n=1).present(ITEM)[0]
    para = ParaphraseWithModel(p, n=1).present(ITEM)[0]
    assert (
        formal.kind == "formality"
        and lexical.kind == "lexical"
        and para.kind == "paraphrase"
    )
    # each carries a genuinely different directive
    assert "casual" in formal.question and "synonym" in lexical.question
    assert len({formal.question, lexical.question, para.question}) == 3
    assert all(isinstance(s, Perturbation) for s in (FormalityShift(p), LexicalSwap(p)))
