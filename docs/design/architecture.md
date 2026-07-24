# WobbleLab architecture

> **Status: design spec, being built toward.** This is the agreed shape of the measurement
> core. Decisions marked *(decided)* were made to keep momentum and are open to revision.

## The core idea

WobbleLab is **sensitivity analysis of a model's output with respect to knobs that shouldn't
move it** (or whose mattering we want to quantify). Every measurement picks a point in
**(model × harness × knob-values)** space, runs it, and reports either **the anchor value** or a
**deviation from the anchor**. "Wobble" is how much the output moves as a nuisance knob turns.

Two things make a measurement trustworthy, and both were the thing missing before:

1. **Name the anchor first.** The anchor is a fully specified operating point. Every number is
   the anchor, or a deviation from it. Nothing is measured in a regime the benchmark is never
   actually run in.
2. **Everything is one knob away from the anchor.** A knob is a transform that changes exactly
   one dimension of the anchor harness and holds the rest fixed. That is what makes a movement
   attributable to a cause.

### Two lenses, two anchors

The same machinery serves two questions with **different anchors**, and they must not be mixed:

- **Benchmark validity** — "is the leaderboard number real, or an artifact of harness choices?"
  Anchor = **the authors' canonical harness** (their exact prompt, shots, scoring, extraction,
  sampling, budget). Perturb the *harness* knobs. Wobble is a property of the benchmark.
- **Production reliability** — "will this model be consistent for *my* use?" Anchor = **the
  user's real operating point** (their prompt, temp > 0). Perturb with meaning-preserving
  changes and reruns. Wobble is a property of the model in that use.

---

## Non-negotiable principles

- **The validation handshake.** Running a benchmark at its canonical anchor must approximately
  reproduce the authors' published score for a known model. If it does not, we are not measuring
  the benchmark, and the tool says so instead of dressing up wobble numbers. This is a hard gate.
- **Coverage is signal.** Parse-rate, truncation-rate, refusal-rate, format-compliance are
  first-class outputs. A metric computed from low-coverage outputs is noise wearing a number's
  clothes and is flagged as such automatically.
- **Attribution.** Every reported movement names its cause: model, harness, or benchmark. A knob
  that moves the number is the reason we can make that attribution.
- **Reproducibility.** Aggregation is pure and order-deterministic (same numbers as a sequential
  run). Every seed, config, and knob value is recorded. The audit JSON reproduces the run.

---

## Data model

```
Provider          the model behind an OpenAI-compatible endpoint (already built)
Item              one benchmark question: id, question, options, correct answer
Harness           the full, immutable recipe for turning an Item into an Outcome
Benchmark         Item source + its CanonicalHarness + published reference scores
Knob              a named transform  Harness -> [Harness]  (deviate one dimension)
Outcome           one (harness, item, seed) result, provenance-tagged
Study             orchestrates anchor + knob variants over items -> Outcomes -> metrics
```

### `Harness` — the recipe (the object that was missing)

```
Harness = {
  prompt:   PromptSpec    # template text, option format "(A)" vs "A.", shots, system prompt
  scoring:  ScoringSpec   # "gen" | "ll" ; extraction regex cascade
  sampling: SamplingSpec  # temperature, top_p, top_k, penalties, max_tokens, stop, seed policy
  order:    OrderSpec     # option order: canonical | correct-at-slot(i) | shuffle(seed)
}
```

A harness is immutable. A knob produces a new one with `replace(...)`.

### `Benchmark` — data + canonical harness + reference *(decided: config + thin subclass)*

Most of the canonical harness is **declarative** (dataset id/split, prompt template, extraction
patterns, sampling, budget, letter set) and lives in a per-benchmark **config** (TOML). The parts
that genuinely vary are **code**: mapping dataset rows to `Item`s (including the option-shuffle
policy), and assembling few-shot exemplars (e.g. MMLU-Pro's per-category CoT examples from the
validation split). So a `Benchmark` is a small subclass that fills a `CanonicalHarness` from
config and implements those two hooks.

It also stores **published reference scores** per known model. Those power the validation
handshake.

### `Knob` — a transform, tagged

```
Knob:
  name             str
  family           "harness" | "perturbation" | "systems" | "task"
  meaning_preserving  bool   # does turning it change the correct answer?
                             #   True  -> movement is a DEFECT
                             #   False -> movement is CHOICE-DEPENDENCE
  variants(canonical: Harness) -> list[(value, Harness)]
  preconditions(h: Harness)    -> list[str]   # warn on incoherent combos
```

`preconditions` is what stops nonsense: a budget knob under `ll` scoring, a rerun knob asserting
determinism it can't have, and so on.

### Sweep and wobble are the same machinery

- A **sweep** is a knob run across a *range of values* (budget in `[4 .. 4096]`) -> a curve.
- A **wobble test** is a *perturbation* knob measured as deviation from the anchor.

Both are "apply knob, run over items, aggregate." They differ only in aggregation, so one engine
serves both, over any `model × benchmark × knob × value`.

### `Study` — the one entry point

```
run_study(provider, benchmark, knobs) ->
    anchor  = base_run(benchmark.canonical, items)      # + validation-handshake gate
    studies = [ execute(h, items) for knob in knobs for (val, h) in knob.variants(canonical) ]
    -> aggregate relative to anchor -> one audit JSON
```

`execute` is the pure concurrent engine already built. Everything it emits is provenance-tagged.

---

## The base run (always multi-run, even at temp 0)

The anchor is not a single pass. **The base run executes the canonical harness N times (N > 1),
at the canonical temperature.** The observational axis is always measured; temperature decides
what the run-to-run variance isolates.

- **At temp > 0**, run-to-run variance is sampling noise (over whatever systems floor is below).
- **At temp 0**, sampling contributes nothing, so any run-to-run difference is *pure systems
  nondeterminism*. Temp 0 is the clean setting for isolating that floor, but it only appears when
  the batch composition varies (below).

**Where the systems floor comes from.** GPU matmul/attention kernels are generally deterministic
run-to-run for a *bit-identical* batch, but they are not **batch-invariant**: the result for your
sequence depends on the batch size and composition it was processed in, because the kernel's
reduction/tiling strategy changes with batch shape and floating-point reduction is
non-associative. The same prompt run alone vs. alongside 31 other requests can produce slightly
different logits; at a near-tied greedy token that flips the argmax and can cascade into a
divergent completion. In a live server this reads as run-to-run nondeterminism because concurrent
load varies the batch. (See Thinking Machines, "Defeating Nondeterminism in LLM Inference," 2025.)

**So how we measure it.** A plain temp-0 replay that re-sends the same items at the same
concurrency may batch near-identically and show ~0 variance. That *understates* the floor, it
does not prove it is absent. To measure it we deliberately **vary the batch composition**: a
batch-composition knob that runs the same item alone vs. in batches of increasing size (or under
varying concurrency) and reports how often the greedy output flips. That is the controlled
isolate. A plain N-times replay catches the floor only to the extent that continuous-batching
timing jitter varies the batches on its own, so the batch-composition knob, not identical
replays, is the real measurement.

The anchor score is the mean across the N base runs; the run-to-run spread is reported alongside
it as observational wobble at the canonical temperature.

---

## Knob taxonomy

Every entry is a `Harness -> [Harness]` transform; the architecture treats them uniformly.

**A. Harness config** *(family: harness, meaning_preserving: False — moving these is a choice)*
scoring method (gen / ll / first-token / constrained) · token budget · CoT vs direct · few-shot
count and which exemplars · prompt wording/style · option delimiter format · extraction
strictness · system prompt · sampling config (top_p / top_k / min_p / penalties) · quantization.

**B. Meaning-preserving perturbation** *(family: perturbation, meaning_preserving: True — moving
these is a defect)* rerun · typed paraphrase (formal / casual / lexical / hedge) · translation /
cross-lingual · option reorder · irrelevant-context or typo injection · relevant-info position
(lost-in-the-middle) · persona framing · demanded output format (JSON vs prose vs letter).

**C. Systems nondeterminism** *(family: systems)* batch-composition flip at temp 0 (see base
run) · seed variation at temp > 0 · cross-backend (vLLM vs llama.cpp vs hosted) · precision /
hardware.

**D. Task-shaped elasticity** *(family: task)* knife-edge probing (minimal rephrase that flips a
yes/no) · ranking stability (Kendall-tau between runs) · recommendation consistency under
paraphrase · numeric-answer spread · code: behavior-hash (test-pass) stability under rephrase or
rerun.

**E. Meta / aggregate** wobble vs **scale** (model capability as the x-axis) · wobble vs
**difficulty** (is it wobbly on things it should know, or only on genuinely hard items?) · wobble
by **domain**.

Highest-value / most distinctive: the **temp-0 batch-nondeterminism floor** (nobody reports it),
the **canonical-vs-production temperature gap**, **wobble-vs-scale**, and **wobble-vs-difficulty**
(which separates appropriate uncertainty from a real defect — the general form of the decidability
gate, D-007).

---

## Temperature

Most modern benchmarks specify **temp 0** for the point number, deliberately, for reproducibility
(MMLU-Pro: temp 0; GPQA baseline: temp 0, with 0.7 only for self-consistency). Consequences:

- The canonical number lives at a determinism setting **production never uses**. The
  canonical-vs-production temperature gap is a headline in its own right.
- The observational axis at temp 0 is not zero in practice (see the base run); it is the
  nondeterminism floor, and it is measured.
- Temperature is a first-class knob on both lenses: a harness choice for benchmark validity, and
  the defining operating point for production reliability.

---

## The benchmark lens: named harnesses as landmarks

The unlock: **a named real harness is just a point in knob-space.** lm-eval's GPQA *is*
`{0-shot, no-CoT, MC-log-likelihood, prompt-A}`; Inspect's GPQA *is* `{0-shot, CoT, gen+extract,
"ANSWER: $LETTER", 4 epochs}`; the authors' *is* `{0-shot, CoT, gen+extract, "(X)", temp 0}`. So
the "named harnesses" and the "sweep between knob values" are the *same space*. The harnesses are
landmarks; the sweep fills in between them. (Concrete catalog: `research/harness-comparison.md`.)

### Harnesses are configs in one engine, not black-box wraps

The primary mechanism is to **re-implement each named harness as a `Harness` config in our own
uniform engine**, not to invoke lm-eval / HELM / Inspect as external runners. Three reasons:
those frameworks each own the model-calling, have hostile dependencies, and emit different output
shapes; you cannot *sweep between* three black boxes because there is no shared parameterization
to interpolate across; and our systems knobs (batch variance, seed, backend) only compose if we
control the model interface. Invoking the real framework is an **optional fidelity check**, not
the run engine.

The price is the **validation handshake**, per (harness × benchmark): our re-implementation of a
harness must reproduce that harness's *published number* for a known model, within tolerance. This
is the fidelity gate that keeps the re-implementation honest, and it is the main cost of the lens
(the LLaMA-65B example — 15 points from a `Choices:` prefix and an extraction change — is the
warning that fidelity must be exact). Credibility lives in the handshake.

### Landmarks + one-knob-at-a-time, not the dense grid

The full cross-product of knobs is combinatorially large *and* full of incoherent cells (CoT + MC-
log-likelihood does not type-check; CoT needs generation). Do not run it. The two valuable designs:

- **Landmark points** — the 3-4 real named harnesses. "Here are the numbers people actually cite
  and their spread." This is the one-model-many-numbers product directly.
- **One-knob-at-a-time (OAT) from the anchor** — start at the authors' config, turn *one* knob,
  measure the delta. "CoT is worth +X, 5-shot +Y, this extraction +Z." Decomposes the wobble into
  per-knob contributions and stays inside valid, coherent configs (`Knob.variants(anchor)`).

Reserve the dense grid for an opt-in research mode over a chosen 2-3 knobs when you specifically
want interaction effects.

### Two knob layers

- **Harness-config knobs** (CoT, shots, format, scoring) define *which harness you are* — the
  landmark + OAT layer.
- **Systems / perturbation knobs** (batch variance, seed, backend, paraphrase) are applied *at* a
  chosen harness (usually the anchor), holding the config fixed, and measure stability *there*.

So a run is layered: pick a harness config (anchor or a landmark), then apply systems/perturbation
knobs on top. "Authors' settings as the anchor, then batch-variance and custom knobs on top" is
exactly this.

### What a benchmark run produces

1. **Landmark numbers** — the model under each real named harness, each pegged to its knob values,
   each passing its handshake.
2. **The OAT decomposition** — per-knob deltas from the authors' anchor, ranked (the hierarchy).
3. **The systems floor** — batch/seed variance at the anchor (the base run).
4. → the **benchmark-side reliability card**: e.g. "Model X on GPQA: 49% at the authors' anchor;
   0.31-0.72 across four real harnesses; scoring-paradigm dominates at ±15pt; temp-0 batch floor
   ±1pt; vs Model Y the 3-point leaderboard gap does / does not survive."

Everything here is the `Study` layer orchestrating `Harness` configs and `Knob.variants`; it is
not new machinery, it is the data model applied to the benchmark lens.

---

## The audit JSON

One self-describing object per study. Someone should be able to reproduce the run and verify
every number from this file alone.

```
run_id, wobblelab_version, environment{python, os, concurrency, gpu?}
started_at, ended_at, runtime_seconds
provider   { model, backend_label, sampling_defaults }         # secrets and IPs sanitized OUT
benchmark  { name, variant, dataset_id, split, n_items, item_shuffle_seed, reference_scores }
canonical  { full Harness spec: prompt_template, shots, scoring, extraction, temp, budget, stop }
anchor     { metric, value, ci, coverage, n_runs, run_to_run_spread,
             reference_delta, gates{handshake, coverage, truncation} }
studies[]  { knob, family, meaning_preserving, values[],
             results[]{ value, metric, coverage, ci, runtime_seconds, started_at, ended_at },
             runtime_seconds }
wobble_summary { per-knob magnitude, the hierarchy (ranked), combined score }
call_log_ref   # optional path to per-call log: prompt_hash, seed, raw_output, parsed, truncated
```

Runtimes and timestamps at every level (per-call optional, per-study, total). Configs, knobs,
values, params, all present.

---

## Gates (automated, non-negotiable)

- **Handshake gate** — anchor value within tolerance of a stored reference score for a known
  model, else flag "harness likely not canonical."
- **Coverage gate** — parse/format coverage below threshold flags the number "suspect."
- **Truncation gate** — fraction of outputs hitting `max_tokens` above threshold flags "budget
  too small for this model under this prompt." (This is the sweep discipline, D-011, made
  structural: the operating point must be a plateau, not a cliff edge.)

---

## Decisions *(made to keep momentum; revisit freely)*

- **Config + thin subclass** for benchmarks. TOML for the declarative harness spec; a small
  subclass for row→Item mapping and few-shot assembly.
- **Named harnesses are `Harness` configs in our engine, not black-box wraps.** Re-implement
  lm-eval / HELM / Inspect / authors' harnesses as configs in the uniform space; invoking the
  real framework is an optional fidelity check. Each carries a published reference number, and the
  handshake (reproduce it) is the gate.
- **Benchmark sweeps = landmarks + OAT from the anchor, not the dense grid.** Run the real named
  harnesses plus one-knob-at-a-time deviations from the authors' anchor; dense factorial is
  opt-in research mode.
- **Library core first, experiments after.** Build `Harness` / `Knob` / `Benchmark` / `Study`
  in `src/wobblelab/`, then make `experiments/` thin callers.
- **Benchmark-validity lens first.** It is where the reference-score handshake lives, and that
  handshake disciplines everything else. Production lens second.
- **Multi-run N default** for the base run: start at N = 10 (tunable), so the temp-0
  nondeterminism floor is always visible.

---

## First vertical slice

Prove the architecture on one end-to-end path before adding breadth. **GPQA Diamond, three
landmark harnesses, one known model.**

1. `Benchmark` for GPQA Diamond, with three `Harness` configs: the **authors' reference**
   (free-gen, `(A)` options, "The correct answer is (X)", extraction cascade, temp 0, plateau
   budget), **lm-eval / Open LLM Leaderboard** (0-shot MC-log-likelihood, `acc_norm`), and
   **Inspect** (CoT, "ANSWER: $LETTER", epochs).
2. Run a known model under each; **handshake**: each landmark's number must land near its
   published figure. If it does, we have three trustworthy numbers and their spread — the
   one-model-many-numbers product, on one benchmark.
3. Base run (N reruns) at the authors' anchor for the temp-0 systems floor.
4. Then OAT knobs from the anchor: reorder, temperature, budget, CoT-on/off, shots.

---

## Open questions (for when rested)

- Reference-score sourcing: where we pull published numbers per benchmark, and the handshake
  tolerance.
- How much per-call logging to keep by default vs on demand (audit depth vs file size).
- The combined "wobble score" definition across families (the D-007 score generalized to the
  full knob set).
- Adaptive sampling to spend calls where variance is high, instead of fixed N over all items.
