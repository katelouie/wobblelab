"""OpenAI-compatible provider: point squishlab at any `/v1` endpoint.

vLLM, llama.cpp's `llama-server`, mlx-lm's server, ollama's `/v1`, or a hosted API all speak
the OpenAI chat-completions protocol, so *one* provider reaches all of them. Combined with
`evaluate(concurrency=N)`, the server batches the concurrent requests -- which is where the
throughput (and the entire point of renting a GPU) comes from. A single-stream client on an
A100 is barely faster than the laptop; a concurrent client feeding vLLM is 50-100x.

Sampling defaults mirror `CONTROLLED` (pure temperature, full softmax) in OpenAI param names.
Backend-specific knobs (vLLM's `top_k` / `min_p` / `repetition_penalty`) go through
`extra_body`, since they aren't standard OpenAI fields.
"""

from __future__ import annotations

import requests

from squishlab.benchmark import LETTERS

DEFAULT_URL = "http://localhost:8000/v1"

# Controlled, portable sampling in OpenAI names (temperature is the only live knob).
CONTROLLED_OAI = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4}


class OpenAICompatibleProvider:
    """A synchronous Provider for any OpenAI-compatible chat-completions endpoint.

    The concurrency lives in `evaluate(concurrency=N)`, not here -- this stays a plain sync
    client so it's identical to reason about and drop-in for `OllamaClient`. Point `base_url`
    at a local server (`http://localhost:8000/v1`) or a rented pod, pass `api_key` for hosted.
    """

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_URL,
        api_key: str | None = None,
        options: dict | None = None,
        extra_body: dict | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.options = {**CONTROLLED_OAI, **(options or {})}
        self.extra_body = extra_body or {}
        self.timeout = timeout
        # One pooled session, sized for concurrency (many in-flight requests across threads).
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def ask(self, prompt: str, seed: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": seed,
            **self.options,
            **self.extra_body,
        }
        return self._post(payload)["choices"][0]["message"]["content"] or ""

    def rank_letters(
        self, prompt: str, n_options: int, seed: int = 0, top_logprobs: int = 20
    ) -> tuple[int | None, dict[str, float]]:
        """Log-likelihood readout via chat-completions `logprobs` (see OllamaClient for the
        rationale). Deterministic (temperature 0, one token); returns the chosen presentation
        position and the per-letter logprobs, letters outside the returned top-N treated as -inf.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": seed,
            "logprobs": True,
            "top_logprobs": min(
                top_logprobs, 20
            ),  # OpenAI caps at 20; 20 holds ~99.9% mass
            "temperature": 0.0,
            "max_tokens": 1,
            "top_p": self.options.get("top_p", 1.0),
            **self.extra_body,
        }
        choice = self._post(payload)["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        candidates = LETTERS[:n_options]
        scores: dict[str, float] = {}
        if content:
            for entry in content[0].get("top_logprobs", []):
                tok = entry["token"].strip()
                if tok in candidates and tok not in scores:
                    scores[tok] = entry["logprob"]
        if not scores:
            return None, {}
        chosen = max(scores, key=scores.__getitem__)
        return candidates.index(chosen), scores

    def config(self) -> dict:
        return {
            "provider": "openai-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "options": self.options,
            "extra_body": self.extra_body,
        }
