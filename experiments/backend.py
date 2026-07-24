"""One place to resolve the model backend, so local <-> remote is an env flip, not an edit.

Every runner builds its provider through `make_provider()`. By default it points at the local
llama.cpp server (`http://localhost:8080/v1`, model `qwen3`); set the `WOBBLE_*` env vars to
aim the exact same experiment at a rented vLLM pod instead. See `docs/remote-gpu.md`.

    # local (default): nothing to set
    python experiments/run_gpqa.py

    # remote vLLM pod (e.g. RunPod), through an SSH tunnel on :8000:
    export WOBBLE_BASE_URL=http://localhost:8000/v1
    export WOBBLE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    export WOBBLE_MODEL_LABEL="Qwen2.5-7B-Instruct"
    export WOBBLE_BACKEND="vLLM (RunPod 4090)"
    export WOBBLE_API_KEY=...            # matches vllm --api-key
    python experiments/run_gpqa.py
"""

from __future__ import annotations

import os

from wobblelab import OpenAICompatibleProvider

# What the /v1 endpoint expects vs. what we print. Local llama.cpp serves qwen3:0.6b under the
# bare name "qwen3"; a vLLM pod serves under the full HF id. LABEL is cosmetic (plots/output).
MODEL_API = os.environ.get("WOBBLE_MODEL", "qwen3")
MODEL_LABEL = os.environ.get("WOBBLE_MODEL_LABEL") or os.environ.get(
    "WOBBLE_MODEL", "qwen3:0.6b"
)
BASE_URL = os.environ.get("WOBBLE_BASE_URL", "http://localhost:8080/v1")
BACKEND_LABEL = os.environ.get("WOBBLE_BACKEND", "llama.cpp --parallel 8")
API_KEY = os.environ.get("WOBBLE_API_KEY") or None


def make_provider(
    *, options: dict | None = None, extra_body: dict | None = None
) -> OpenAICompatibleProvider:
    """Build the configured provider. Qwen3 is a thinking model, so its `<think>` block is
    suppressed for clean single-token `ll` / short `gen`; non-thinking models (Qwen2.5, most
    others) don't get the chat-template kwarg, which they may not support."""
    eb = dict(extra_body or {})
    if "qwen3" in MODEL_API.lower():
        eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
    return OpenAICompatibleProvider(
        MODEL_API,
        base_url=BASE_URL,
        api_key=API_KEY,
        options=options,
        extra_body=eb,
    )
