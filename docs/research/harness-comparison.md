# How one model gets many benchmark numbers

A catalog of how the popular harnesses and leaderboards actually run **GPQA Diamond** and
**MMLU-Pro**, with sample scores. This is the benchmark-lens product in miniature: these are the
real, named harnesses whose spread WobbleLab quantifies. **Inspect AI** (UK AISI) is included and
called out because it is what Anaconda uses internally for our model offerings, so it is a
first-class anchor for us.

Two honest caveats up front:
- Clean *same-model, cross-harness* numbers are hard to assemble; most published scores are the
  *same benchmark* run under *different* conditions on *different* model sets. The one clean
  controlled example (LLaMA-65B on MMLU) is below.
- The aggregators themselves say scores within 2-3 points are a tie and are "directional, not
  absolute" ([IntuitionLabs](https://intuitionlabs.ai/articles/gpqa-diamond-ai-benchmark)). That
  concession *is* the thesis.

---

## GPQA Diamond — how each harness runs it

| Harness | Shots | CoT | Scoring | Answer format | Runs | Notes |
|---|---|---|---|---|---|---|
| **Authors' reference** ([idavidrein/gpqa](https://github.com/idavidrein/gpqa)) | 0 (also few-shot CoT variants) | optional | free-gen + regex | `The correct answer is (X)` | 1 (self-consistency at temp 0.7) | temp 0 for the single answer; `(A) ... (D)` options |
| **lm-eval / Open LLM Leaderboard v2** ([HF](https://github.com/huggingface/leaderboards)) | **0** | **no** | **multiple-choice log-likelihood**, `acc_norm` (random=0.25 → 0) | pick highest-likelihood letter | 1 | compresses scores toward chance; a different *paradigm*, not a tweak |
| **Inspect AI** (AISI / Anaconda) ([inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)) | 0 | **yes** (default `cot=True`) | free-gen + extract, simple accuracy | `ANSWER: $LETTER` | **epochs default 4**, per-question averaged | opinionated dataset→Task→Solver→Scorer; provider default tool config |
| **HELM** ([Stanford CRFM](https://github.com/stanford-crfm/helm)) | scenario-defined | scenario | generation + match | scenario-defined | 1 + **perturbations** | reports **worst-case accuracy** over typo / dialect perturbations |
| **Provider-reported** (e.g. Meta) | 0-shot CoT | yes | free-gen CoT | bespoke | 1+ | optimized prompting; not the authors' reference harness |
| **Independent aggregators** (Artificial Analysis, vals.ai) | own scaffold | yes | own | own | own | standardized but partly opaque methodology |

The dominant divergence on GPQA is the **scoring paradigm**: 0-shot multiple-choice
log-likelihood (Open LLM Leaderboard) vs. CoT free-generation (Inspect, providers, authors). That
alone moves a capable model 15-20+ points, because 0-shot MC-LL on graduate questions sits far
below what the same model does when allowed to reason.

## MMLU-Pro — how each harness runs it

| Harness | Shots | CoT | Scoring | Budget | Notes |
|---|---|---|---|---|---|
| **Authors' reference** ([TIGER-Lab](https://github.com/TIGER-AI-Lab/MMLU-Pro)) | **5** | **yes** | free-gen + regex `answer is (X)` | 2048 tok, stop `Question:` | temp 0; 10 options (A-J) |
| **lm-eval / Open LLM Leaderboard v2** | **5** | yes | gen + extract, normalized | — | closest to the authors' of the leaderboards |
| **Inspect AI** | configurable (commonly 5) | yes | free-gen + extract, accuracy | — | same primitives as its GPQA task |
| **Artificial Analysis** | own | tested CoT vs direct | own | — | methodology page gated; CoT helps |

MMLU-Pro's harnesses agree more than GPQA's (most are 5-shot CoT-gen), so the cross-harness spread
is smaller here than on GPQA — but the CoT-vs-direct choice still moves it 6-19 points (the
MMLU-Pro authors' own figure).

---

## Sample scores

### The clean controlled example: MMLU, LLaMA-65B

The one case with the *same model* under *named different harnesses*
([HF writeup](https://huggingface.co/blog/open-llm-leaderboard-mmlu)):

| Implementation | MMLU |
|---|---|
| Original | **0.637** |
| HELM | **0.637** |
| Eleuther Harness (Jan 2023) | **0.488** |

**~15 points** on one model, purely from prompt formatting and answer-extraction differences.

### Llama 3.1 (Meta-reported, CoT) vs. the leaderboard paradigm

Meta's own numbers ([eval_details](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md)),
all CoT:

| Model | GPQA Diamond (0-shot CoT) | MMLU-Pro (5-shot CoT) |
|---|---|---|
| Llama 3.1 405B | 49.0 | 73.3 |
| Llama 3.1 70B | 50.5 | 68.9 |

Two things to note. First, 70B *outscores* 405B on GPQA Diamond (50.5 vs 49.0) — that inversion is
inside the noise of a 198-item set, a wobble illustration in the published numbers themselves.
Second, these CoT numbers are far above what the **Open LLM Leaderboard v2** paradigm (0-shot
MC-log-likelihood, `acc_norm`) reports for the same models, where GPQA scores compress toward the
chance floor. (Exact leaderboard figures move; check the live board rather than trust a cached
number — itself a small lesson.)

### GPQA Diamond via Inspect AI (Anaconda's harness), frontier models

Inspect's own baselines (CoT, 198 items, 1 epoch)
([inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)):

| Model | GPQA Diamond |
|---|---|
| Gemini 3 Pro | 0.929 |
| Claude Sonnet 4.5 | 0.717 |
| GPT-5.1 | 0.652 |
| Human expert baseline | 0.697 |

### Public GPQA / MMLU-Pro leaderboards (2026, mixed scaffolds)

Directional only, different scaffolds per source:
GPQA Diamond top: Gemini 3.1 Pro ~94.1, MiniMax M3 ~92.9, Claude Fable 5 ~92.6
([pricepertoken](https://pricepertoken.com/leaderboards/benchmark/gpqa)). MMLU-Pro top: Gemini 3
Pro ~89.8, Claude Opus 4.5 ~89.5 ([Artificial Analysis](https://artificialanalysis.ai/evaluations/mmlu-pro)).

---

## What this means for WobbleLab

- **These named harnesses are the benchmark-lens product.** "Run this model under lm-eval's
  0-shot-MC GPQA vs. Inspect's CoT GPQA vs. the authors' free-gen GPQA, and report the spread" is
  a credible harness-wobble study using harnesses people actually use — no synthetic perturbation
  needed.
- **The GPQA divergence is dominated by scoring paradigm** (MC-log-likelihood vs. CoT-gen), which
  is a 15-20+ point axis. That is the single biggest harness knob for this benchmark.
- **Anchor on Inspect AI for the Anaconda story.** Since Anaconda evaluates model offerings with
  Inspect, our handshake and reliability card should reproduce Inspect's GPQA/MMLU-Pro method
  (CoT, `ANSWER: $LETTER`, epochs-averaged) and then show what the number does under the other
  named harnesses.
- **Epochs are a built-in multi-run.** Inspect already runs `epochs` (default 4) per question,
  which is exactly the observational/multi-run axis — a natural hook for our temp-0
  nondeterminism-floor measurement.
