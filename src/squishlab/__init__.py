"""squishlab -- measuring how much LLM answers move under things that shouldn't matter."""

from squishlab.client import CONTROLLED, OllamaClient
from squishlab.squish import model_squish, squish_score
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
    "wilson_ci",
    "newcombe_diff_ci",
    "confident_shift",
    "bootstrap_ci",
    "squish_score",
    "model_squish",
]
