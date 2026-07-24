"""Backend throughput / concurrency bake-off: ollama vs llama.cpp vs mlx-lm (F-024).

Same model on all three, a fixed MC load fired at concurrency 1/4/8/16, reporting req/s so
you can see each backend's concurrency scaling. `/no_think` suppresses Qwen3's thinking trace
so we time answer generation, not reasoning length.

Finding: llama.cpp `--parallel` wins (batches + logprobs + works through our adapter); ollama
auto-batches to ~4 (zero setup); mlx_lm.server does NOT batch (flat). See lab-journal F-024.

Start the two servers first (ollama is already running):

    # llama.cpp on the SAME gguf ollama uses (identical model/quant):
    GGUF=$(ollama show --modelfile qwen3:0.6b | awk '/^FROM \\//{print $2}')
    llama-server -m "$GGUF" --port 8080 -c 8192 --parallel 8 --jinja -ngl 99 &

    # mlx-lm (8-bit; note the quant differs from the gguf):
    mlx_lm.server --model mlx-community/Qwen3-0.6B-8bit --port 8081 &

Then: python experiments/backends.py
"""

import time
from concurrent.futures import ThreadPoolExecutor

from wobblelab import OllamaClient, OpenAICompatibleProvider
from wobblelab.loaders import load_mmlu

N_PROMPTS = 40
CONCURRENCY = (1, 4, 8, 16)


def mk_prompt(item) -> str:
    lines = [item.question, ""]
    for i, opt in enumerate(item.options):
        lines.append(f"{chr(65 + i)}. {opt}")
    lines += ["", "Answer with only the letter (A/B/C/D). /no_think"]
    return "\n".join(lines)


def timed(provider, prompts, concurrency) -> float:
    t = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(lambda p: provider.ask(p, seed=0), prompts))
    return time.time() - t


def main() -> None:
    prompts = [
        mk_prompt(it) for it in load_mmlu("world_religions", n=N_PROMPTS, seed=1)
    ]
    providers = {
        "ollama": OllamaClient("qwen3:0.6b", options={"num_predict": 8}),
        "llama.cpp": OpenAICompatibleProvider(
            "qwen3", base_url="http://localhost:8080/v1", options={"max_tokens": 8}
        ),
        "mlx-lm": OpenAICompatibleProvider(
            "mlx-community/Qwen3-0.6B-8bit",
            base_url="http://localhost:8081/v1",
            options={"max_tokens": 8},
        ),
    }
    print(f"load: {len(prompts)} MC prompts\n")
    print(f"{'backend':10} {'conc':>4} {'sec':>7} {'req/s':>7} {'speedup':>8}")
    print("-" * 42)
    for name, prov in providers.items():
        try:
            prov.ask(prompts[0], 0)  # warmup
            base = None
            for c in CONCURRENCY:
                dt = timed(prov, prompts, c)
                base = base or dt
                print(
                    f"{name:10} {c:>4} {dt:>7.1f} {len(prompts) / dt:>7.1f} {base / dt:>7.1f}x"
                )
        except Exception as e:
            print(f"{name:10} ERROR: {type(e).__name__}: {str(e)[:70]}")
        print()


if __name__ == "__main__":
    main()
