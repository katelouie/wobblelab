"""Tests for the Task seam: the two-method contract every benchmark family implements."""

import pytest

from squishlab import (
    MockProvider,
    MultipleChoiceTask,
    Outcome,
    Task,
    picks_option_containing,
)
from squishlab.benchmark import MCItem
from squishlab.task import CodeExecutionTask, JudgeTask

ITEM = MCItem(
    id="t", question="pick it", options=("CORRECT", "w1", "w2", "w3"), answer_idx=0
)


def test_mc_task_satisfies_the_task_protocol():
    assert isinstance(MultipleChoiceTask("ll"), Task)
    assert MultipleChoiceTask("ll").deterministic is True
    assert MultipleChoiceTask("gen").deterministic is False


def test_mc_task_rejects_bad_scoring():
    with pytest.raises(ValueError):
        MultipleChoiceTask("logprobs")


def test_mc_run_scores_and_locates_the_answer():
    task = MultipleChoiceTask("ll")
    provider = MockProvider(picks_option_containing("CORRECT"))
    # One perturbation per slot; the correct answer sits at slot p in perturbation p.
    for p, order in enumerate(task.perturbations(ITEM)):
        out = task.run(provider, ITEM, order, seed=0)
        assert isinstance(out, Outcome)
        assert out.correct_slot == p  # answer placed at slot p
        assert out.slot == p  # a perfect model finds it there
        assert out.content == ITEM.answer_idx  # chose the answer's ORIGINAL index
        assert out.correct is True
        assert out.n_slots == 4


def test_mc_run_content_is_order_invariant_for_a_perfect_model():
    # The whole point of `content`: same answer content picked, whatever the slot it moved to.
    task = MultipleChoiceTask("ll")
    provider = MockProvider(picks_option_containing("CORRECT"))
    contents = {
        task.run(provider, ITEM, order, 0).content for order in task.perturbations(ITEM)
    }
    assert contents == {
        ITEM.answer_idx
    }  # never flips content -> zero interventional squish


def test_planned_seams_raise_until_built():
    for task in (CodeExecutionTask(), JudgeTask()):
        with pytest.raises(NotImplementedError):
            task.perturbations(ITEM)
        with pytest.raises(NotImplementedError):
            task.run(None, ITEM, None, 0)
