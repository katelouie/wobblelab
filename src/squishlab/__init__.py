"""squishlab -- measuring how much LLM answers move under things that shouldn't matter."""

from squishlab.client import CONTROLLED, OllamaClient
from squishlab.stats import confident_shift, newcombe_diff_ci, wilson_ci

__version__ = "0.0.1"

__all__ = [
    "OllamaClient",
    "CONTROLLED",
    "wilson_ci",
    "newcombe_diff_ci",
    "confident_shift",
]
