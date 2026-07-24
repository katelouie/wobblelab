"""The full wobble gauntlet on GPQA Diamond (198 graduate-level MCQs) via llama.cpp.

GPQA Diamond is hard on purpose: PhD experts in-field score ~65%, skilled non-experts
with web access ~34%. A 0.6B model lives near the 25% chance floor, which is exactly
what makes it a clean wobble probe -- when accuracy is near chance, the reported number
is almost pure position-bias artifact, and the gauntlet says so out loud.

Pass 1 (reorder, ll): position-debiased accuracy + bootstrap CI + accuracy at each of the
four answer slots + reorder wobble (how often a meaning-preserving option shuffle flips the
answer) -- Pillar 2, and how it propagates into the headline number.
Pass 2 (natural order, gen, 5 runs): the leaderboard-style accuracy and its run-to-run
variance -- Pillar 1, the sampling-noise axis. 198 items is small, so expect a wider
run-to-run band than a 12k corpus would show.

Backend: llama-server --parallel 8 on qwen3:0.6b (llama.cpp can't load qwen3.5), through the
OpenAI adapter with enable_thinking=false (clean single-token ll + short gen). concurrency=8.

    GGUF=$(ollama show --modelfile qwen3:0.6b | awk '/^FROM \\//{print $2}')
    llama-server -m "$GGUF" --port 8080 -c 32768 --parallel 8 --jinja -ngl 99 &
    python experiments/run_gpqa.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wobblelab import (  # noqa: E402
    MultipleChoiceTask,
    NaturalOrder,
    OpenAICompatibleProvider,
    evaluate,
    score_stability,
)
from wobblelab.benchmark import LETTERS  # noqa: E402
from wobblelab.loaders import load_gpqa_diamond  # noqa: E402

BENCHMARK = "GPQA-Diamond"
MODEL = "qwen3:0.6b"
BACKEND = "llama.cpp --parallel 8"
N_ITEMS = None  # all 198
N_RUNS = 5
CONCURRENCY = 8


def plot(report, stab, path):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.patch.set_facecolor("#12141a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#181b23")
        for s in ax.spines.values():
            s.set_color("#3a3f4b")
        ax.tick_params(colors="#8a8f9a")

    # Pillar 2: accuracy by answer position
    by = report.accuracy_by_position
    k = len(by)
    ax1.bar(range(k), by, color="#f0a63c", width=0.7)
    ax1.axhline(
        1 / k, color="#7ec98f", lw=1.4, ls=(0, (4, 3)), label=f"chance {1 / k:.0%}"
    )
    ax1.axhline(
        report.accuracy,
        color="#8fb8d8",
        lw=1.4,
        label=f"debiased {report.accuracy:.0%}",
    )
    ax1.set_xticks(range(k))
    ax1.set_xticklabels(list(LETTERS[:k]), color="#c9b79e", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("accuracy", color="#c9b79e")
    ax1.set_title(
        f"PILLAR 2 · position bias (swing {report.position_swing:.2f})",
        color="#f6e8ce",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=8
    )

    # Pillar 1: run-to-run accuracy
    runs = stab["runs"]
    ax2.axhline(
        stab["mean"],
        color="#8fb8d8",
        lw=1.5,
        label=f"mean {stab['mean']:.1%} ± {stab['std'] * 100:.1f}pt",
    )
    ax2.bar(range(len(runs)), runs, color="#7ec98f", width=0.6)
    ax2.set_xticks(range(len(runs)))
    ax2.set_xticklabels(
        [f"run {i + 1}" for i in range(len(runs))], color="#c9b79e", fontsize=8
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("natural-position accuracy", color="#c9b79e")
    ax2.set_title(
        f"PILLAR 1 · run-to-run (spread {stab['spread'] * 100:.1f}pt @ {report.n_items} items)",
        color="#f6e8ce",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=8
    )

    fig.suptitle(
        f"WobbleLab · full {BENCHMARK} · {MODEL} (llama.cpp)",
        color="#f6e8ce",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main():
    t0 = time.time()
    prov = OpenAICompatibleProvider(
        "qwen3",
        base_url="http://localhost:8080/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    items = load_gpqa_diamond(N_ITEMS)
    print(f"loaded {len(items)} GPQA-Diamond items")

    # Pass 1: reorder-ll (Pillar 2 + propagation to overall/debiased accuracy)
    t = time.time()
    report = evaluate(
        prov, items, scoring="ll", benchmark="gpqa-diamond", concurrency=CONCURRENCY
    )
    p1 = time.time() - t
    print(f"\n[pass 1 · reorder-ll · {p1 / 60:.1f} min]")
    print(
        f"  debiased accuracy {report.accuracy:.3f}  CI[{report.accuracy_ci[0]:.3f},{report.accuracy_ci[1]:.3f}]"
    )
    print(
        f"  reorder wobble {report.wobble_by_kind['reorder']:.3f}  position swing {report.position_swing:.3f}"
    )
    print(f"  accuracy by slot: {[round(a, 3) for a in report.accuracy_by_position]}")

    # Pass 2: natural-order multi-run gen (Pillar 1 · run-to-run variance)
    t = time.time()
    natural = MultipleChoiceTask("gen", perturbations=[NaturalOrder()])
    stab = score_stability(
        prov, items, n_runs=N_RUNS, task=natural, n_rerun=1, concurrency=CONCURRENCY
    )
    p2 = time.time() - t
    print(f"\n[pass 2 · natural-position gen × {N_RUNS} · {p2 / 60:.1f} min]")
    print(
        f"  leaderboard accuracy {stab['mean']:.3f}  run-to-run std {stab['std']:.3f}  spread {stab['spread']:.3f}"
    )
    print(f"  runs: {[round(x, 3) for x in stab['runs']]}")

    print("\n=== propagation ===")
    print(
        f"  leaderboard (natural, gen):  {stab['mean']:.1%}  ± {stab['std'] * 100:.1f}pt run-to-run"
    )
    print(
        f"  debiased    (reorder, ll):   {report.accuracy:.1%}  [{report.accuracy_ci[0]:.1%},{report.accuracy_ci[1]:.1%}]"
    )
    print(
        f"  hidden by the single number: {report.position_swing:.0%} position swing, {report.wobble_by_kind['reorder']:.0%} reorder flip"
    )

    results = Path(__file__).resolve().parent.parent / "results"
    results.mkdir(exist_ok=True)
    plot(report, stab, results / "gpqa_diamond_wobble.png")
    out = {
        "benchmark": BENCHMARK,
        "model": MODEL,
        "backend": BACKEND,
        "n_items": len(items),
        "report": report.to_dict(),
        "stability": stab,
        "minutes": round((time.time() - t0) / 60, 1),
        "pass_minutes": {
            "reorder_ll": round(p1 / 60, 1),
            "natural_gen": round(p2 / 60, 1),
        },
    }
    (results / "gpqa_diamond_wobble.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nwrote results/gpqa_diamond_wobble.{{png,json}} in {out['minutes']} min total"
    )


if __name__ == "__main__":
    main()
