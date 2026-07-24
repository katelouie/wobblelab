"""Harness wobble: how far a benchmark *number* moves when you change the eval harness.

The official MMLU number for this model was produced one specific way -- log-likelihood
scoring (argmax over the model's P("A")/P("B")/... at the first token) with few-shot
exemplars. Our F-017 number was produced another way -- sample a letter, regex it back,
zero-shot. Neither choice changes what the model *knows*; both change the number it
reports. That difference is harness wobble, and here we quantify it by holding the item
set fixed and sweeping two harness axes:

  scoring:  gen  (sample + parse a letter -- our path, exposed to output-side bias)
            ll   (argmax over first-token letter logprobs -- the official path, no sampling)
  shots:    0-shot   vs   5-shot (exemplars from MMLU's purpose-built `dev` split)

That's a 2x2 grid of accuracy on identical items. The spread across cells is the harness
wobble on the reported number; the main effects decompose it into scoring vs shots. A
secondary panel asks whether ll scoring cures the 48-point position bias gen showed
(F-017) -- run at 0-shot, where we can compare directly to the benchmark run.

Same 40 items / SEED as experiments/bench.py, so the gen/0-shot cell cross-checks F-017.
Run: python experiments/harness.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wobblelab import OllamaClient, bootstrap_ci  # noqa: E402
from wobblelab.benchmark import (  # noqa: E402
    LETTERS,
    MCItem,
    format_prompt,
    orders_correct_at_each_position,
    parse_answer,
)

MODEL = "qwen3.5:0.8b"
QUANT = "Q8_0"
SUBJECT = "world_religions"
N_ITEMS = 40
N_RERUN = 5  # gen only; ll is deterministic (temp 0)
N_SHOTS = 5
SEED = 0

# Official Qwen3.5-0.8B, non-thinking mode (HF model card, F-019). A LOOSE external
# reference only: it's the MMLU-Redux *aggregate* over all subjects, log-likelihood +
# few-shot + full precision + the full benchmark -- not our single 40-item subject.
OFFICIAL_MMLU_REDUX = 0.485

client = OllamaClient(MODEL, options={"num_predict": 8})


def load_mmlu_with_dev(subject: str, n: int, seed: int):
    """Test items (sampled) + the 5 canonical dev exemplars for few-shot prompting."""
    from datasets import load_dataset

    def to_item(r, j):
        return MCItem(
            id=f"{subject}:{j}",
            question=r["question"],
            options=tuple(r["choices"]),
            answer_idx=int(r["answer"]),
        )

    test = load_dataset("cais/mmlu", subject, split="test")
    idx = list(range(len(test)))
    random.Random(seed).shuffle(idx)
    items = [to_item(test[j], j) for j in idx[:n]]

    dev = load_dataset("cais/mmlu", subject, split="dev")
    exemplars = [to_item(dev[j], f"dev{j}") for j in range(min(N_SHOTS, len(dev)))]
    return items, exemplars


def exemplar_block(item: MCItem) -> str:
    """One few-shot exemplar: the question in natural order, then its answer letter."""
    order = tuple(range(len(item.options)))
    return f"{format_prompt(item, order)}\n{LETTERS[item.answer_idx]}\n\n"


# --- the two scoring readouts, uniform signature: (prompt, n, seed) -> (pos, n_seen) ---
# n_seen is the count of candidate letters ll actually saw in the top-20 (None for gen),
# so we can report how lossy ollama's top-20 logprob truncation is on real items.


def choose_gen(prompt: str, n: int, seed: int) -> tuple[int | None, int | None]:
    return parse_answer(client.ask(prompt, seed=seed), n), None


def choose_ll(prompt: str, n: int, seed: int) -> tuple[int | None, int | None]:
    pos, scores = client.rank_letters(prompt, n, seed=seed)
    return pos, len(scores)


def natural_accuracy(items, choose, preamble: str, rerun: int):
    """Accuracy with each answer at its *dataset* position -- the leaderboard number."""
    per_item, seen_full, seen_tot = [], 0, 0
    for it in items:
        n = len(it.options)
        prompt = preamble + format_prompt(
            it, tuple(range(n))
        )  # natural order: pos==idx
        hits = 0
        for s in range(rerun):
            pos, n_seen = choose(prompt, n, s)
            if n_seen is not None:
                seen_tot += 1
                seen_full += n_seen == n
            hits += pos == it.answer_idx
        per_item.append(hits / rerun)
    coverage = (seen_full / seen_tot) if seen_tot else None  # ll top-20 completeness
    return per_item, coverage


def position_profile(items, choose, preamble: str, rerun: int):
    """Place the answer at every slot -> per-position accuracy + chosen-letter bias."""
    k = len(items[0].options)
    pos_hits, pos_tot, chosen = [0] * k, [0] * k, [0] * k
    for it in items:
        n = len(it.options)
        for p, order in enumerate(orders_correct_at_each_position(it)):
            prompt = preamble + format_prompt(it, order)
            for s in range(rerun):
                pos, _ = choose(prompt, n, s)
                if pos is not None:
                    chosen[pos] += 1
                pos_tot[p] += 1
                pos_hits[p] += pos == p
    acc_by_pos = [h / t for h, t in zip(pos_hits, pos_tot)]
    total = sum(chosen) or 1
    return acc_by_pos, [c / total for c in chosen]


def plot(grid, profiles, path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    fig.patch.set_facecolor("#12141a")
    order = [("gen", "0shot"), ("ll", "0shot"), ("gen", "5shot"), ("ll", "5shot")]
    colors = {"gen": "#f0a63c", "ll": "#7ec98f"}

    # left: the 2x2 grid as grouped bars, with CIs + the official reference line
    ax1.set_facecolor("#181b23")
    xs = range(len(order))
    accs = [grid[c]["acc"] for c in order]
    errs = [
        [grid[c]["acc"] - grid[c]["ci"][0] for c in order],
        [grid[c]["ci"][1] - grid[c]["acc"] for c in order],
    ]
    ax1.bar(
        xs,
        accs,
        width=0.62,
        color=[colors[s] for s, _ in order],
        yerr=errs,
        ecolor="#c9b79e",
        capsize=4,
    )
    for x, (s, sh) in zip(xs, order):
        ax1.text(
            x,
            grid[(s, sh)]["acc"] + 0.03,
            f"{grid[(s, sh)]['acc']:.2f}",
            ha="center",
            color="#e7dcc8",
            fontsize=9,
        )
    ax1.axhline(0.25, color="#666", lw=1, ls=(0, (4, 3)), label="chance (0.25)")
    ax1.axhline(
        OFFICIAL_MMLU_REDUX,
        color="#8fb8d8",
        lw=1.4,
        ls=(0, (2, 2)),
        label=f"official MMLU-Redux {OFFICIAL_MMLU_REDUX:.2f} (loose ref)",
    )
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([f"{s}\n{sh}" for s, sh in order], color="#c9b79e")
    ax1.set_ylim(0, 0.75)
    ax1.set_ylabel("natural-position accuracy", color="#c9b79e")
    ax1.set_title(
        "harness wobble: same items, four harnesses", color="#f6e8ce", fontsize=11
    )
    ax1.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=7.5
    )

    # right: does ll cure gen's position bias? (0-shot). answer placed at each slot.
    ax2.set_facecolor("#181b23")
    k = len(profiles["gen"]["acc_by_pos"])
    letters = list(LETTERS[:k])
    w = 0.38
    for i, s in enumerate(("gen", "ll")):
        prof = profiles[s]["acc_by_pos"]
        swing = max(prof) - min(prof)
        ax2.bar(
            [x + (i - 0.5) * w for x in range(k)],
            prof,
            width=w,
            color=colors[s],
            label=f"{s}  (swing {swing:.2f})",
        )
    ax2.axhline(0.25, color="#666", lw=1, ls=(0, (4, 3)))
    ax2.set_xticks(range(k))
    ax2.set_xticklabels(letters, color="#c9b79e")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("accuracy", color="#c9b79e")
    ax2.set_xlabel(
        "position of the correct answer  (flat = no position bias)", color="#c9b79e"
    )
    ax2.set_title(
        "0-shot: position bias by scoring method", color="#f6e8ce", fontsize=11
    )
    ax2.legend(
        facecolor="#181b23", edgecolor="#3a3f4b", labelcolor="#e7dcc8", fontsize=8
    )

    fig.suptitle(
        f"HARNESS WOBBLE · MMLU:{SUBJECT} · {MODEL} ({QUANT})",
        color="#f6e8ce",
        fontsize=13,
        fontweight="bold",
    )
    for ax in (ax1, ax2):
        for sp in ax.spines.values():
            sp.set_color("#3a3f4b")
        ax.tick_params(colors="#8a8f9a")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")


def main() -> None:
    t0 = time.time()
    items, exemplars = load_mmlu_with_dev(SUBJECT, N_ITEMS, SEED)
    few = "".join(exemplar_block(e) for e in exemplars)

    cells = {
        ("gen", "0shot"): (choose_gen, "", N_RERUN),
        ("ll", "0shot"): (choose_ll, "", 1),
        ("gen", "5shot"): (choose_gen, few, N_RERUN),
        ("ll", "5shot"): (choose_ll, few, 1),
    }
    grid = {}
    for cell, (choose, preamble, rerun) in cells.items():
        per_item, coverage = natural_accuracy(items, choose, preamble, rerun)
        acc = sum(per_item) / len(per_item)
        ci = bootstrap_ci(per_item, lambda s: sum(s) / len(s), n_boot=4000, seed=7)
        grid[cell] = {
            "acc": round(acc, 3),
            "ci": [round(x, 3) for x in ci],
            "coverage": coverage,
        }
        cov = "n/a" if coverage is None else f"{coverage:.3f}"
        print(
            f"  {cell[0]:>3}/{cell[1]}: acc {acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  ll-coverage {cov}"
        )

    # position-bias profiles at 0-shot: the mechanism behind the gen number
    profiles = {}
    for s, choose, rerun in (("gen", choose_gen, N_RERUN), ("ll", choose_ll, 1)):
        acc_by_pos, chosen = position_profile(items, choose, "", rerun)
        profiles[s] = {
            "acc_by_pos": [round(a, 3) for a in acc_by_pos],
            "chosen_dist": [round(c, 3) for c in chosen],
            "swing": round(max(acc_by_pos) - min(acc_by_pos), 3),
            "debiased_acc": round(sum(acc_by_pos) / len(acc_by_pos), 3),
        }

    # decompose the harness wobble into main effects on the reported (natural) number
    order = [("gen", "0shot"), ("ll", "0shot"), ("gen", "5shot"), ("ll", "5shot")]
    accs = {c: grid[c]["acc"] for c in order}
    mean = lambda cs: sum(accs[c] for c in cs) / len(cs)  # noqa: E731
    scoring_effect = mean([c for c in order if c[0] == "ll"]) - mean(
        [c for c in order if c[0] == "gen"]
    )
    shot_effect = mean([c for c in order if c[1] == "5shot"]) - mean(
        [c for c in order if c[1] == "0shot"]
    )
    harness_wobble = max(accs.values()) - min(accs.values())

    print(f"\nHARNESS WOBBLE · MMLU:{SUBJECT} · {MODEL} ({QUANT}) · {len(items)} items")
    print(
        f"  spread across the four harnesses: {harness_wobble:.3f}  ({min(accs.values()):.3f} -> {max(accs.values()):.3f})"
    )
    print(f"  main effect  scoring (ll - gen): {scoring_effect:+.3f}")
    print(f"  main effect  shots  (5 - 0):     {shot_effect:+.3f}")
    print(
        f"  0-shot position swing:  gen {profiles['gen']['swing']:.3f}  vs  ll {profiles['ll']['swing']:.3f}"
    )
    print(
        f"  0-shot debiased acc:    gen {profiles['gen']['debiased_acc']:.3f}  vs  ll {profiles['ll']['debiased_acc']:.3f}"
    )
    print("  (cross-check: gen/0-shot debiased should ~ F-017's 0.364)")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot(grid, profiles, results_dir / f"harness_{SUBJECT}.png")
    out = {
        "model": MODEL,
        "quant": QUANT,
        "subject": SUBJECT,
        "config": client.config(),
        "n_items": len(items),
        "n_rerun_gen": N_RERUN,
        "n_shots": N_SHOTS,
        "official_mmlu_redux": OFFICIAL_MMLU_REDUX,
        "grid": {f"{s}_{sh}": grid[(s, sh)] for s, sh in order},
        "profiles_0shot": profiles,
        "harness_wobble": round(harness_wobble, 3),
        "scoring_effect": round(scoring_effect, 3),
        "shot_effect": round(shot_effect, 3),
        "seconds": round(time.time() - t0, 1),
    }
    (results_dir / f"harness_{SUBJECT}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/harness_{SUBJECT}.png and .json in {out['seconds']}s")


if __name__ == "__main__":
    main()
