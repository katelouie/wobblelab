"""squishlab -- measuring how much LLM answers move under things that shouldn't matter."""

from squishlab.client import CONTROLLED, OllamaClient
from squishlab.provider import (
    MockProvider,
    Provider,
    always_position,
    picks_option_containing,
)
from squishlab.perturb import (
    FormalityShift,
    LexicalSwap,
    ParaphraseWithModel,
    Perturbation,
    Presentation,
    RephraseInstruction,
    ReorderOptions,
    TranslateWithModel,
)
from squishlab.report import (
    ModelReport,
    compare,
    compare_markdown,
    evaluate,
    score_stability,
)
from squishlab.squish import model_squish, squish_factor, squish_score
from squishlab.task import MultipleChoiceTask, Outcome, Task
from squishlab.stats import (
    bootstrap_ci,
    confident_shift,
    newcombe_diff_ci,
    wilson_ci,
)

__version__ = "0.0.1"

__all__ = [
    "OllamaClient",
    "CONTROLLED",
    "Provider",
    "MockProvider",
    "always_position",
    "picks_option_containing",
    "Task",
    "Outcome",
    "MultipleChoiceTask",
    "Perturbation",
    "Presentation",
    "ReorderOptions",
    "RephraseInstruction",
    "ParaphraseWithModel",
    "FormalityShift",
    "LexicalSwap",
    "TranslateWithModel",
    "ModelReport",
    "evaluate",
    "compare",
    "compare_markdown",
    "score_stability",
    "wilson_ci",
    "newcombe_diff_ci",
    "confident_shift",
    "bootstrap_ci",
    "squish_score",
    "squish_factor",
    "model_squish",
]
