"""wobblelab -- measuring how much LLM answers move under things that shouldn't matter."""

from wobblelab.cards import (
    JSONRenderer,
    Panel,
    ReliabilityCard,
    SVGCardRenderer,
    benchmark_card,
    production_card,
)
from wobblelab.client import CONTROLLED, OllamaClient
from wobblelab.openai_provider import OpenAICompatibleProvider
from wobblelab.provider import (
    MockProvider,
    Provider,
    always_position,
    picks_option_containing,
)
from wobblelab.perturb import (
    FormalityShift,
    LexicalSwap,
    ParaphraseWithModel,
    NaturalOrder,
    Perturbation,
    Presentation,
    RephraseInstruction,
    ReorderOptions,
    TranslateWithModel,
)
from wobblelab.report import (
    ModelReport,
    compare,
    compare_markdown,
    evaluate,
    score_stability,
)
from wobblelab.wobble import model_wobble, wobble_factor, wobble_score
from wobblelab.task import MultipleChoiceTask, Outcome, Task
from wobblelab.stats import (
    bootstrap_ci,
    confident_shift,
    newcombe_diff_ci,
    wilson_ci,
)

__version__ = "0.1.0"

__all__ = [
    "OllamaClient",
    "OpenAICompatibleProvider",
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
    "NaturalOrder",
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
    "wobble_score",
    "wobble_factor",
    "model_wobble",
    "ReliabilityCard",
    "Panel",
    "benchmark_card",
    "production_card",
    "SVGCardRenderer",
    "JSONRenderer",
]
