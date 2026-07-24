r"""GPQA harness sweep: reproduce the official protocol, then wobble it.

F-026 found our default MC harness collapses to 3.4% on GPQA -- not because the model is
anti-correlated with truth, but because `max_tokens=4` + "answer with only the letter" never
lets a tiny model that reflexively explains actually emit a parseable answer. Before trusting
any gen number, we reproduce the *real* harness from github.com/idavidrein/gpqa and then move
the two knobs it leaves implicit.

The official zero-shot protocol (baselines/utils.py, verbatim):
    What is the correct answer to this question: {question}

    Choices:
    (A) {c1}
    (B) {c2}
    (C) {c3}
    (D) {c4}

    Format your response as follows: "The correct answer is (insert answer here)"
Answer extraction is a regex cascade, first valid A-D wins (baselines/run_baseline.py):
    [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']

Two harness knobs, crossed:
  - FORMAT: our terse "answer with only the letter" vs the canonical "The correct answer is (X)".
  - BUDGET: max_tokens the model gets to comply. The canonical format needs room to even *say*
    the answer; the terse one pretends it doesn't.

Pass A (greedy, single pass): accuracy + parse coverage vs budget, canonical vs terse.
Pass B (sampled, 5 runs at the plateau budget): the *corrected* Pillar-1 run-to-run variance,
replacing F-026's degenerate 3.4%.

Backend: llama-server --parallel 8 on qwen3:0.6b, OpenAI adapter, enable_thinking=false.

    GGUF=$(ollama show --modelfile qwen3:0.6b | awk '/^FROM \\//{print $2}')
    llama-server -m "$GGUF" --port 8080 -c 32768 --parallel 8 --jinja -ngl 99 &
    python experiments/gpqa_harness_sweep.py
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backend import BACKEND_LABEL, MODEL_LABEL, make_provider  # noqa: E402
from wobblelab.benchmark import LETTERS, format_prompt, parse_answer  # noqa: E402
from wobblelab.loaders import load_gpqa_diamond  # noqa: E402

MODEL = MODEL_LABEL
CONCURRENCY = 8
BUDGETS = [4, 8, 16, 32, 64, 128, 256]
PLATEAU_BUDGET = 256  # for the corrected variance pass
N_RUNS = 5

# The official GPQA extraction cascade (idavidrein/gpqa), first valid A-D wins.
GPQA_PATTERNS = [
    r"answer is \((.)\)",
    r"Answer: \((.)\)",
    r"answer: \((.)\)",
    r"answer \((.)\)",
    r"\((.)\)",
]


def gpqa_prompt(item) -> str:
    """The canonical zero-shot GPQA prompt, options in the item's (already-shuffled) order."""
    c = item.options
    body = f"What is the correct answer to this question: {item.question}"
    body += "\n\nChoices:\n" + "\n".join(
        f"({LETTERS[i]}) {c[i]}" for i in range(len(c))
    )
    body += '\n\nFormat your response as follows: "The correct answer is (insert answer here)"'
    return body


def extract_gpqa(text: str, n: int) -> int | None:
    """Regex cascade; return the 0-based letter index, or None if nothing valid parses."""
    valid = LETTERS[:n]
    for pat in GPQA_PATTERNS:
        m = re.search(pat, text)
        if m and m.group(1).upper() in valid:
            return valid.index(m.group(1).upper())
    return None


def _map(fn, items):
    """Concurrent map preserving order (server batches the in-flight requests)."""
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        return list(ex.map(fn, items))


def greedy_pass(items, prompt_fn, extract_fn, budget) -> dict:
    """One deterministic pass: accuracy + parse coverage under a given format + budget."""
    prov = make_provider(options={"temperature": 0.0, "max_tokens": budget})

    def one(item):
        raw = prov.ask(prompt_fn(item), seed=0)
        pos = extract_fn(raw, len(item.options))
        return (pos, item.answer_idx)

    got = _map(one, items)
    parsed = [p for p, _ in got if p is not None]
    correct = sum(1 for p, a in got if p == a)
    return {
        "budget": budget,
        "accuracy": round(correct / len(items), 4),
        "coverage": round(
            len(parsed) / len(items), 4
        ),  # fraction that emitted a letter
    }


def sampled_run(items, prompt_fn, extract_fn, budget, seed) -> float:
    """One sampled pass (temp 1.0) at a fixed seed; accuracy over the natural order."""
    prov = make_provider(options={"temperature": 1.0, "max_tokens": budget})

    def one(item):
        raw = prov.ask(prompt_fn(item), seed=seed)
        return extract_fn(raw, len(item.options)) == item.answer_idx

    return round(sum(_map(one, items)) / len(items), 4)


def terse_prompt(item) -> str:
    return format_prompt(item, tuple(range(len(item.options))))


def main():
    t0 = time.time()
    items = load_gpqa_diamond()
    print(f"loaded {len(items)} GPQA-Diamond items\n")

    # Pass A: budget sweep under the canonical harness (greedy).
    print("=== Pass A · canonical harness, budget sweep (greedy) ===")
    canonical = []
    for b in BUDGETS:
        r = greedy_pass(items, gpqa_prompt, extract_gpqa, b)
        canonical.append(r)
        print(
            f"  budget {b:4d}: accuracy {r['accuracy']:.3f}  parse-coverage {r['coverage']:.3f}"
        )

    # Contrast: the terse letter-only harness (our default) at a small and a large budget.
    print("\n=== contrast · terse letter-only harness (greedy) ===")
    terse = []
    for b in (4, 64):
        r = greedy_pass(items, terse_prompt, parse_answer, b)
        terse.append(r)
        print(
            f"  budget {b:4d}: accuracy {r['accuracy']:.3f}  parse-coverage {r['coverage']:.3f}"
        )

    # Pass B: corrected Pillar-1 variance at the plateau budget, canonical harness, sampled.
    print(
        f"\n=== Pass B · corrected run-to-run (canonical, budget {PLATEAU_BUDGET}, "
        f"sampled × {N_RUNS}) ==="
    )
    runs = [
        sampled_run(items, gpqa_prompt, extract_gpqa, PLATEAU_BUDGET, seed=s)
        for s in range(N_RUNS)
    ]
    mean = sum(runs) / len(runs)
    var = sum((x - mean) ** 2 for x in runs) / len(runs)
    std = var**0.5
    spread = max(runs) - min(runs)
    print(f"  runs: {runs}")
    print(f"  mean {mean:.3f}  std {std:.3f}  spread {spread:.3f}")

    # ---- what the harness knobs did to the number ----
    acc_by_budget = {r["budget"]: r["accuracy"] for r in canonical}
    canon_swing = max(acc_by_budget.values()) - min(acc_by_budget.values())
    print("\n=== harness swing on GPQA ===")
    print(
        f"  canonical, budget alone: {min(acc_by_budget.values()):.3f} -> "
        f"{max(acc_by_budget.values()):.3f}  ({canon_swing:.3f} spread)"
    )
    print(
        f"  terse letter-only @4:    {terse[0]['accuracy']:.3f} "
        f"(coverage {terse[0]['coverage']:.3f})  <- F-026's collapse"
    )
    print(
        f"  canonical @{PLATEAU_BUDGET}:          {acc_by_budget[PLATEAU_BUDGET]:.3f} "
        f"(coverage {canonical[-1]['coverage']:.3f})"
    )

    results = Path(__file__).resolve().parent.parent / "results"
    results.mkdir(exist_ok=True)
    plot(canonical, terse, runs, mean, std, results / "gpqa_harness_sweep.png")
    out = {
        "benchmark": "GPQA-Diamond",
        "model": MODEL,
        "backend": BACKEND_LABEL,
        "n_items": len(items),
        "canonical_by_budget": canonical,
        "terse_contrast": terse,
        "corrected_variance": {
            "budget": PLATEAU_BUDGET,
            "runs": runs,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "spread": round(spread, 4),
        },
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (results / "gpqa_harness_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/gpqa_harness_sweep.{{png,json}} in {out['minutes']} min")


def plot(canonical, terse, runs, mean, std, path):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.patch.set_facecolor("#12141a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#181b23")
        for s in ax.spines.values():
            s.set_color("#3a3f4b")
        ax.tick_params(colors="#8a8f9a")

    # Panel 1: accuracy + coverage vs token budget (canonical)
    budgets = [r["budget"] for r in canonical]
    acc = [r["accuracy"] for r in canonical]
    cov = [r["coverage"] for r in canonical]
    ax1.plot(budgets, acc, "o-", color="#f0a63c", lw=2, label="accuracy (canonical)")
    ax1.plot(
        budgets, cov, "s--", color="#8fb8d8", lw=1.6, label="parse coverage", alpha=0.9
    )
    ax1.axhline(0.25, color="#7ec98f", lw=1.2, ls=(0, (4, 3)), label="chance 25%")
    ax1.scatter(
        [4],
        [terse[0]["accuracy"]],
        color="#e05a5a",
        zorder=5,
        s=55,
        label=f"terse letter-only @4 ({terse[0]['accuracy']:.0%})",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(b) for b in budgets], color="#c9b79e", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("max_tokens budget", color="#c9b79e")
    ax1.set_ylabel("fraction", color="#c9b79e")
    ax1.set_title(
        "PILLAR 0 · the number depends on the budget",
        color="#f6e8ce",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=7.5
    )

    # Panel 2: corrected run-to-run at the plateau budget
    ax2.axhline(
        mean, color="#8fb8d8", lw=1.5, label=f"mean {mean:.1%} ± {std * 100:.1f}pt"
    )
    ax2.axhline(0.25, color="#7ec98f", lw=1.2, ls=(0, (4, 3)), label="chance 25%")
    ax2.bar(range(len(runs)), runs, color="#7ec98f", width=0.6)
    ax2.set_xticks(range(len(runs)))
    ax2.set_xticklabels(
        [f"run {i + 1}" for i in range(len(runs))], color="#c9b79e", fontsize=8
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("natural-order accuracy", color="#c9b79e")
    ax2.set_title(
        f"PILLAR 1 · corrected run-to-run (budget {PLATEAU_BUDGET})",
        color="#f6e8ce",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=8
    )

    fig.suptitle(
        f"WobbleLab · GPQA-Diamond harness sweep · {MODEL} ({BACKEND_LABEL})",
        color="#f6e8ce",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")


if __name__ == "__main__":
    main()
