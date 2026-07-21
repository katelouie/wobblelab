"""Benchmark squish on MMLU: accuracy WITH a CI, plus an option-reordering squish.

For each item we place the correct answer at *every* option position (distractors keep
their relative order) and rerun each. That yields:
  - a position-DEBIASED accuracy, with a CI bootstrapped over items (the correct unit --
    reruns of one item are correlated, so a naive Wilson over trials would lie);
  - accuracy broken out by answer position -> the position-bias impact on the score;
  - option-reorder interventional squish (does the chosen answer's CONTENT change as we
    shuffle the options?) -- guaranteed meaning-preserving, so pure squish;
  - the raw chosen-letter distribution.

Controlled config, seeded, reuses squishlab.stats. Run: python experiments/bench.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from squishlab import OllamaClient, bootstrap_ci, wilson_ci  # noqa: E402
from squishlab.benchmark import (  # noqa: E402
    LETTERS,
    MCItem,
    format_prompt,
    modal,
    orders_correct_at_each_position,
    parse_answer,
    presented_to_original,
)

MODEL = "qwen3.5:0.8b"
QUANT = "Q8_0"
SUBJECT = "world_religions"
N_ITEMS = 40
N_RERUN = 5
SEED = 0

client = OllamaClient(MODEL, options={"num_predict": 8})


def load_mmlu(subject: str, n: int, seed: int) -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", subject, split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for j in idx[:n]:
        r = ds[j]
        items.append(
            MCItem(
                id=f"{subject}:{j}",
                question=r["question"],
                options=tuple(r["choices"]),
                answer_idx=int(r["answer"]),
            )
        )
    return items


def measure(item: MCItem):
    """Return per-item metrics + global-accumulation contributions."""
    k = len(item.options)
    orders = orders_correct_at_each_position(item)  # index p -> correct sits at pos p
    modal_content = []  # modal chosen ORIGINAL index, per order
    correct_hits = 0
    total = 0
    obs = []  # per-order rerun instability
    pos_correct = [0] * k  # correct-count when answer is at position p
    chosen_letter = [0] * k
    for p, order in enumerate(orders):
        chosen_orig, chosen_pos = [], []
        for s in range(N_RERUN):
            text = client.ask(format_prompt(item, order), seed=s)
            pos = parse_answer(text, k)
            chosen_pos.append(pos)
            chosen_orig.append(presented_to_original(pos, order))
            if pos is not None:
                chosen_letter[pos] += 1
            if pos == p:  # correct answer sits at position p
                pos_correct[p] += 1
                correct_hits += 1
            total += 1
        mc, frac = modal(chosen_orig)
        modal_content.append(mc)
        obs.append(1 - frac)
    _, content_frac = modal(modal_content)
    interventional = round(1 - content_frac, 3)  # 0 = same content across all orders
    observational = round(sum(obs) / len(obs), 3)
    return (
        {
            "id": item.id,
            "accuracy": round(correct_hits / total, 3),
            "interventional": interventional,
            "observational": observational,
            "squish": round(max(interventional, observational), 3),
        },
        pos_correct,
        chosen_letter,
        k,
    )


def plot(acc_by_pos, acc_mean, acc_ci, chosen_dist, path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6), dpi=150, gridspec_kw={"width_ratios": [3, 2]}
    )
    for f in (fig,):
        f.patch.set_facecolor("#12141a")
    k = len(acc_by_pos)
    letters = list(LETTERS[:k])
    # left: accuracy by answer-position, with debiased mean + CI band
    ax1.set_facecolor("#181b23")
    ax1.axhspan(acc_ci[0], acc_ci[1], color="#8fb8d8", alpha=0.18, zorder=0)
    ax1.axhline(
        acc_mean,
        color="#8fb8d8",
        lw=1.5,
        zorder=1,
        label=f"debiased acc {acc_mean:.2f} [{acc_ci[0]:.2f},{acc_ci[1]:.2f}]",
    )
    ax1.bar(letters, acc_by_pos, color="#f0a63c", width=0.6, zorder=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("accuracy", color="#c9b79e")
    ax1.set_xlabel(
        "position of the correct answer  (flat = no position bias)", color="#c9b79e"
    )
    ax1.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=8
    )
    # right: chosen-letter distribution (which slot the model reaches for)
    ax2.set_facecolor("#181b23")
    ax2.bar(letters, chosen_dist, color="#e06666", width=0.6)
    ax2.axhline(1 / k, color="#7ec98f", lw=1.5, ls=(0, (4, 3)), label="unbiased (1/k)")
    ax2.set_ylim(0, max(0.5, max(chosen_dist) * 1.15))
    ax2.set_ylabel("fraction chosen", color="#c9b79e")
    ax2.set_xlabel("letter the model picked", color="#c9b79e")
    ax2.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=8
    )
    fig.suptitle(
        f"BENCHMARK SQUISH · MMLU:{SUBJECT} · {MODEL} ({QUANT})",
        color="#f6e8ce",
        fontsize=13,
        fontweight="bold",
    )
    for ax in (ax1, ax2):
        for s in ax.spines.values():
            s.set_color("#3a3f4b")
        ax.tick_params(colors="#8a8f9a")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    t0 = time.time()
    items = load_mmlu(SUBJECT, N_ITEMS, SEED)
    rows, pos_correct_tot, chosen_tot, k = [], None, None, None
    for it in items:
        r, pc, cl, k = measure(it)
        rows.append(r)
        pos_correct_tot = (
            pc
            if pos_correct_tot is None
            else [a + b for a, b in zip(pos_correct_tot, pc)]
        )
        chosen_tot = (
            cl if chosen_tot is None else [a + b for a, b in zip(chosen_tot, cl)]
        )

    per_item_acc = [r["accuracy"] for r in rows]
    acc_mean = sum(per_item_acc) / len(per_item_acc)
    acc_ci = bootstrap_ci(per_item_acc, lambda s: sum(s) / len(s), n_boot=4000, seed=1)
    naive = wilson_ci(
        round(acc_mean * len(items) * k * N_RERUN), len(items) * k * N_RERUN
    )
    acc_by_pos = [c / (len(items) * N_RERUN) for c in pos_correct_tot]
    chosen_dist = [c / sum(chosen_tot) for c in chosen_tot]
    interv = [r["interventional"] for r in rows]
    interv_mean = sum(interv) / len(interv)
    interv_ci = bootstrap_ci(interv, lambda s: sum(s) / len(s), n_boot=4000, seed=2)
    squish_mean = sum(r["squish"] for r in rows) / len(rows)
    worst = sorted(rows, key=lambda r: r["squish"], reverse=True)[:5]

    print(
        f"BENCHMARK SQUISH · MMLU:{SUBJECT} · {MODEL} ({QUANT}) · {len(items)} items × {k} orders × {N_RERUN}"
    )
    print(
        f"accuracy (debiased): {acc_mean:.3f}  bootstrap95 [{acc_ci[0]:.3f},{acc_ci[1]:.3f}]"
        f"   (naive Wilson-over-trials, too tight: [{naive[0]:.3f},{naive[1]:.3f}])"
    )
    print(
        f"accuracy by answer-position {LETTERS[:k]}: {[round(a, 2) for a in acc_by_pos]}"
        f"   swing = {max(acc_by_pos) - min(acc_by_pos):.2f}"
    )
    print(
        f"chosen-letter distribution: {[round(c, 2) for c in chosen_dist]}  (unbiased = {1 / k:.2f})"
    )
    print(
        f"reorder squish (interventional): {interv_mean:.3f} [{interv_ci[0]:.3f},{interv_ci[1]:.3f}]"
        f"   · mean item squish {squish_mean:.3f}"
    )
    print(
        "worst items:",
        ", ".join(f"{w['id'].split(':')[-1]}({w['squish']})" for w in worst),
    )

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot(
        acc_by_pos, acc_mean, acc_ci, chosen_dist, results_dir / f"bench_{SUBJECT}.png"
    )
    out = {
        "model": MODEL,
        "quant": QUANT,
        "subject": SUBJECT,
        "config": client.config(),
        "n_items": len(items),
        "n_rerun": N_RERUN,
        "orders_per_item": k,
        "accuracy_debiased": round(acc_mean, 3),
        "accuracy_ci_bootstrap": [round(x, 3) for x in acc_ci],
        "accuracy_by_position": [round(a, 3) for a in acc_by_pos],
        "chosen_letter_dist": [round(c, 3) for c in chosen_dist],
        "reorder_squish": round(interv_mean, 3),
        "reorder_squish_ci": [round(x, 3) for x in interv_ci],
        "mean_item_squish": round(squish_mean, 3),
        "seconds": round(time.time() - t0, 1),
        "items": rows,
    }
    (results_dir / f"bench_{SUBJECT}.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nwrote results/bench_{SUBJECT}.png and bench_{SUBJECT}.json in {out['seconds']}s"
    )


if __name__ == "__main__":
    main()
