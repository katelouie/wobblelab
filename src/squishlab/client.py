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

    def config(self) -> dict:
        """The full, loggable config, for the results manifest."""
        return {
            "model": self.model,
            "url": self.url,
            "options": self.options,
            "think": self.think,
        }
