"""The Provider seam: what `evaluate()` needs from a model backend, and nothing more.

A provider answers a prompt two ways -- by *generation* (sample text, the harness parses
a letter out) and by *log-likelihood* (rank the candidate letters directly). Those are the
only two things the benchmark harness ever asks of a model, so they are the whole interface.
`OllamaClient` already satisfies this structurally; a vLLM / OpenAI-compatible provider is a
second implementation of the same three methods (the on-ramp to real models).

`MockProvider` is a deterministic in-memory provider driven by a policy function. It exists
so the harness can be tested (and demoed) with no model running: an "always answers A" policy
must reproduce a position bias, a "knows the answer" policy must score 100%. Fake models keep
the logic honest.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """A model backend the benchmark harness can drive. Three methods, no more."""

    def ask(self, prompt: str, seed: int) -> str:
        """Generation: return the model's text answer (the harness parses a letter)."""
        ...

    def rank_letters(
        self, prompt: str, n_options: int, seed: int = 0, top_logprobs: int = 20
    ) -> tuple[int | None, dict[str, float]]:
        """Log-likelihood: return (chosen presentation position, {letter: logprob})."""
        ...

    def config(self) -> dict:
        """The loggable configuration, for a report's provenance."""
        ...


class MockProvider:
    """A deterministic Provider for tests and demos, with no model behind it.

    `policy(prompt, seed) -> presentation_position | None` decides which lettered option
    the fake model picks. The gen and ll paths agree (ll gives the chosen letter logprob 0
    and omits the rest), so a report is identical across scoring methods for a mock -- which
    is exactly what you want when testing the harness rather than a model.
    """

    def __init__(
        self, policy: Callable[[str, int], int | None], name: str = "mock"
    ) -> None:
        self._policy = policy
        self._name = name

    def ask(self, prompt: str, seed: int) -> str:
        pos = self._policy(prompt, seed)
        return "" if pos is None else _LETTERS[pos]

    def rank_letters(
        self, prompt: str, n_options: int, seed: int = 0, top_logprobs: int = 20
    ) -> tuple[int | None, dict[str, float]]:
        pos = self._policy(prompt, seed)
        if pos is None or pos >= n_options:
            return None, {}
        return pos, {_LETTERS[pos]: 0.0}

    def config(self) -> dict:
        return {"provider": "mock", "name": self._name}


_LETTERS = "ABCDEFGH"


def always_position(pos: int) -> Callable[[str, int], int]:
    """A policy that always picks a fixed slot -- the canonical position-bias fake."""
    return lambda prompt, seed: pos


def picks_option_containing(needle: str) -> Callable[[str, int], int | None]:
    """A policy that reads the rendered options and picks the slot whose text contains
    `needle`. Construct items with the correct option marked, and this is a perfect model."""

    def policy(prompt: str, seed: int) -> int | None:
        for line in prompt.splitlines():
            line = line.strip()
            if (
                len(line) >= 2
                and line[0] in _LETTERS
                and line[1] == "."
                and needle in line
            ):
                return _LETTERS.index(line[0])
        return None

    return policy
