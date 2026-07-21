"""Explicit, portable ollama client for squishlab.

The whole point: NOT ollama's implicit Modelfile defaults. Every sampling knob is
set here to a documented, neutral value so the measurement is reproducible and
comparable across harnesses. See lab-journal D-003.
"""

from __future__ import annotations

import requests

DEFAULT_URL = "http://localhost:11434/api/chat"

# Pure temperature sampling from the full softmax: temperature is the only live
# knob, every truncation and penalty neutralized. Any harness can reproduce this.
CONTROLLED = {
    "temperature": 1.0,  # must be > 0 to observe dispersion at all
    "top_p": 1.0,  # no nucleus truncation
    "top_k": 0,  # disabled
    "min_p": 0.0,  # disabled
    "repeat_penalty": 1.0,  # disabled
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "num_predict": 4,  # we only need one word
}


class OllamaClient:
    """Minimal ollama /api/chat client with an explicit, logged sampling config."""

    def __init__(
        self,
        model: str,
        url: str = DEFAULT_URL,
        options: dict | None = None,
        think: bool = False,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.url = url
        self.options = {**CONTROLLED, **(options or {})}
        self.think = think
        self.timeout = timeout

    def ask(self, prompt: str, seed: int) -> str:
        """One completion. Explicit per-call seed makes the whole run reproducible."""
        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {**self.options, "seed": seed},
                "think": self.think,
                "stream": False,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def rank_letters(
        self, prompt: str, n_options: int, seed: int = 0, top_logprobs: int = 20
    ) -> tuple[int | None, dict[str, float]]:
        """Log-likelihood readout: the first-token logprob of each candidate letter.

        This is how official MC benchmarks score — argmax over the model's probability
        of "A"/"B"/"C"/... as the next token — with no sampling, so it sidesteps the
        output-side biases the generation path is exposed to (F-017). Deterministic
        (temperature 0, one token). Returns the chosen *presentation position* (0-based,
        or None if no candidate letter appears) and the per-letter logprobs; letters
        absent from the returned top_logprobs are simply missing (treated as -inf).

        top_p/top_k stay neutral (CONTROLLED) so the reported distribution is the full
        softmax, not a truncated view.
        """
        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    **self.options,
                    "temperature": 0.0,
                    "num_predict": 1,
                    "seed": seed,
                },
                "think": self.think,
                "logprobs": True,
                "top_logprobs": top_logprobs,
                "stream": False,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        steps = r.json().get("logprobs") or []
        candidates = "ABCDEFGH"[:n_options]
        scores: dict[str, float] = {}
        if steps:
            for entry in steps[0].get("top_logprobs", []):
                tok = entry["token"].strip()
                if tok in candidates and tok not in scores:
                    scores[tok] = entry["logprob"]
        if not scores:
            return None, {}
        chosen = max(scores, key=scores.__getitem__)
        return candidates.index(chosen), scores

    def config(self) -> dict:
        """The full, loggable config, for the results manifest."""
        return {
            "model": self.model,
            "url": self.url,
            "options": self.options,
            "think": self.think,
        }
