# SquishLab — Lab Journal

The decision + discovery record for squishlab. Every methodological fork, why it
went the way it did, and what the data taught us. Append-only, newest context at
the bottom of each section. When a decision is later overturned, don't delete it —
add a superseding entry so the reasoning trail survives.

**squishlab measures *squish*: how much a model's answer moves under things that
should not change it.** Two lenses, which turn out to be two axes:

- **observational** — rerun the *same* prompt; how much does the answer disperse? (sampling noise)
- **interventional** — perturb the prompt in a *meaning-preserving* way; does the answer shift? (fragility)

---

## Decisions ledger

| # | Decision | One-line rationale |
|---|---|---|
| **D-001** | The squish *plane*: dispersion (y) × margin (x), four quadrants (SOLID / KNIFE-EDGE / NOISY-BUT-SURE / COIN-FLIP) | The two lenses are orthogonal; one axis can't tell a solid answer from a knife-edge (see F-001) |
| **D-002** | Model = `qwen3.5:0.8b` at **Q8_0** (8-bit), `think:false` | Quantization is itself a squish source, so 8-bit not 4-bit; no-think avoids reasoning-trace variance polluting the measurement |
| **D-003** | **Controlled, portable sampling config** (pure temperature sampling from the full softmax, explicit per-call seeds) — NOT ollama's Modelfile defaults | Implicit harness defaults are a config-level squish source; "as it ships" measures *ollama's packaging*, not the model, and won't reproduce on vLLM/llama.cpp |
| **D-004** | Interventional margin is **continuous shift-based**; plane default = **worst-case (`margin_max`)** within a prompt; model headline = **central tendency** across prompts; always compute `max`/`mean`/`max_ci`; not a user knob | A knife-edge is *defined* by the existence of a breaking rephrase → worst-case within-prompt; different aggregation levels want different stats (see 2026-07-20 entry) |
| **D-005** | Everything carries CIs (Wilson on rates, Newcombe on shifts). Point for placement, CI for honesty; **CI straddling a threshold → `unresolved`**, not forced into a quadrant. Both axes identical. | A reliability tool must not report point estimates without error bars; N=30 thresholds were slicing sampling noise (F-004) |
| **D-006** | **Typed paraphrase taxonomy** (casual / formal / hedge / reorder / lexical), constant K per battery | Makes the hot-spot diagnostic (*which kind* of change broke it) and keeps worst-case margins comparable |

## Findings ledger

| # | Finding | Where |
|---|---|---|
| **F-001** | In pure binary with only the observational axis, `dispersion = 0.25 − margin²` — the two collapse to one parabola and KNIFE-EDGE is unreachable. The second axis is *necessary*, not decorative. | vPOC-mock |
| **F-002** | The 0.8B's confidence **does not track truth or difficulty**: "water is wet" (trivially true) came back noisy; "is a hotdog a sandwich" (genuinely contested) came back rock-solid. | smoke test |
| **F-003** | **`zero_even` is paraphrase-sensitive.** "Is zero an even number?" → yes (correct), but most meaning-identical rephrasings → no (wrong). A definite math fact is phrasing-dependent and mostly wrong. The gem. | vPOC v0 |
| **F-004** | v0's majority-flip margin was too blunt (7/8 prompts pinned at 1.0); N=30 too few (dispersion threshold split a one-sample-wide cluster). Motivated v0.1. | vPOC v0 |
| **F-005** | **Genuine KNIFE-EDGES exist.** `pluto_planet` (disp 0.02, margin 0.58) and `cereal_soup` (disp 0.07, margin 0.47) are rock-solid on rerun but fragile to paraphrase — the exact phenomenon the two-axis design targets, and invisible to the observational axis alone. | vPOC v0.1 |
| **F-006** | **`reorder` is the model's dominant fragility** (hot-spot for 4/8 prompts). Caveat: our reorder form is a stilted colon-fronting ("A planet: is Pluto one?"), so it's partly a paraphrase-quality artifact — D-006's worst-case/quality caveat, live. | vPOC v0.1 |
| **F-007** | At N=60 / M=25, **all 8 points are `unresolved`** — the 95% CIs are too wide to confidently assign quadrants. The framework correctly refuses to over-claim. Under-powered; needs larger N/M or a continuous/deadzone treatment. | vPOC v0.1 |

---

## 2026-07-18 — design + vPOC-mock

Framed squish as sensitivity to things that shouldn't matter, and split it into the
two lenses above. Built a synthetic mock (`scratchpad/squish_poc.py`) with two
independent knobs per prompt: `true_p_yes` (drives observational dispersion) and
`true_flip_rate` (drives interventional margin).

**F-001 (the reason the plane has two axes):** with only the observational lens in a
pure binary task, `dispersion = 0.25 − vote_margin²` — dispersion and margin are the
*same number* on a parabola, so you can never separate a SOLID answer from a
KNIFE-EDGE (looks-decided-but-flips-on-paraphrase). Adding the interventional axis is
what opens the plane. → **D-001.**

Model plan: a tiny local model so we control everything. 8-bit not 4-bit because
aggressive quantization is *itself* a squish source we don't want to confound with
the model's own behavior. → **D-002.**

## 2026-07-20 — vPOC-real: first light

Model downloaded: `qwen3.5:0.8b`, and `ollama show` confirmed **Q8_0** (the ~1 GB size
was the tell; a Q4 would be ~half). Default temperature 1 (good — we need >0 to see
dispersion), but the Modelfile also ships `presence_penalty 1.5 / top_k 20 / top_p
0.95` and a `thinking` capability.

**Smoke test:** API works, `think:false` suppresses the reasoning trace (message is
just role+content), yes/no parses cleanly. And **F-002**: on 8 reruns, "water is wet"
split 6/2 while "is a hotdog a sandwich" went 8/8 — the model is jittery on the
obvious and confident on the contested. Confidence ≠ difficulty.

**Config decision (D-003):** the smoke ran on ollama's Modelfile defaults. That's a
confound — it measures *ollama's packaging*, and another harness ships different
implicit defaults, so nothing would reproduce or compare. Switched to a fully
explicit, portable config: **pure temperature sampling from the full softmax**
(temp 1, top_p 1, top_k 0, min_p 0, all penalties 0, repeat_penalty 1) with
**explicit per-call seeds 0..N-1** so the whole run is bit-reproducible.

## 2026-07-20 — vPOC v0 (controlled, N=30)

720 controlled calls, 2.5 min. The plane lit up (3/4 quadrants). Two useful outcomes:

- **F-003 (the gem):** `zero_even`. Canonical "Is zero an even number?" → **yes**, but
  three of four rephrasings → **no** (wrong), and the flips don't follow a clean rule
  ("Is 0 even?" → yes, "Is the number zero an even number?" → no). A definite math
  fact comes out phrasing-dependent and mostly *incorrect*. This is the phenomenon
  squishlab exists to catch, found on the first real run, un-designed.
- **F-004 (what v0 got wrong):** (a) the margin metric was majority-flip — it only
  registers a paraphrase that crosses 50%, so it pinned 7/8 prompts at margin 1.0 and
  the whole x-axis collapsed to the right edge. (b) N=30 gives dispersions with wide
  error bars; the 0.15 threshold split `water_wet` (4/30) from the
  `hotdog/prime/tomato` cluster (5/30) — a **one-sample** difference deciding a
  quadrant. Meaningless.

Artifacts: `results/vpoc_real.json`, `results/squish_plane_real.png`.

## 2026-07-20 — the max-vs-mean decision (D-004), worked in full

The v0.1 fix for the blunt x-axis is a *continuous* shift-based margin: per paraphrase,
`δ = p_yes(paraphrase) − p_yes(canonical)`; aggregate the `|δ|`s into one margin. But
*how* to aggregate is the question that defines what "margin" means. Long reasoning,
preserved because it's load-bearing:

**Max and mean are not rival estimators of one truth — they're different quantities.**
`mean|δ|` = average fragility ("random rephrase, how much moves?"). `max|δ|` = peak
fragility ("does there *exist* a breaking rephrase?"). Both legitimate.

**The statistics punish naive max.** Each `|δ_k|` is noisy (~0.13 at M=15). So (a) max
is *noise-inflated* — even with all true shifts zero, the max of K folded-noise values
is biased up; and (b) max is *K-dependent* — more paraphrases → higher max → margin
drifts down as you add paraphrases, which **breaks comparability** (fatal for a
scoring framework; a determinism tool whose own scores wobble with paraphrase count is
a dogfooding failure). Mean has neither problem but **dilutes the signal**: 1 hard flip
in 10 → `mean|δ|`≈0.12 → "robust," and the knife-edge vanishes.

**The mission forces worst-case.** A KNIFE-EDGE is *defined* by the existence of a
breaking rephrase; there is no average-case definition of it. So the within-prompt
aggregation must be worst-case or the axis can't do its one job.

**Resolution (three moves):**
1. *Different aggregation levels want different stats.* **Within a prompt** (paraphrase
   shifts → margin): worst-case. **Across prompts** (per-prompt margins → the model's
   headline squish): central tendency + named worst-offenders. The max-vs-mean fight
   was two questions in one coat.
2. *Tame max with CIs, don't replace it.* Plane plots the worst-case **point**
   (`1 − max|δ̂|`) with its **Newcombe CI** as an error bar, and flags any point whose
   CI straddles the threshold as `unresolved`. Max's noise-inflation becomes a *visible
   uncertainty* instead of a silent lie. (Same treatment as the dispersion axis — D-005.)
3. *Compute all three, always.* Store the full per-paraphrase shift vector; emit
   `margin_max` (plane), `margin_mean` (average fragility), `margin_max_ci`
   (`1 − max` of the CI-lower-bounds — fires only when a shift is statistically
   confident, never on noise), plus the **hot-spot** (argmax paraphrase + its type).
   That *is* "run with both for a while": we run all of them every time and let the
   data pick over the next few batteries.

**Not a user-facing knob.** A `p`-power-mean dial nobody can interpret is a bad
surface; a headline number should mean one thing.

**Two caveats worst-case amplifies:** (1) paraphrase quality is now load-bearing — a
meaning-*changing* paraphrase produces a false squish; mitigations are the auditable
hot-spot every run and the confident-shift requirement. (2) constant K per battery or
comparability dies. → **D-006** (typed, constant-K taxonomy).

## 2026-07-20 — v0.1 built

Refactored the reusable core into the package: `src/squishlab/stats.py` (Wilson +
Newcombe + `confident_shift`, with `tests/test_stats.py`, 6/6 green) and
`src/squishlab/client.py` (the explicit `CONTROLLED` config + `OllamaClient`). The
experiment (`experiments/vpoc_real.py`) now: continuous margins with all three
aggregations + hot-spot; Wilson/Newcombe CIs on both axes; typed paraphrases
(N=60 canonical / 25 per paraphrase); plane with CI error bars and hollow markers for
`unresolved` points. Run launched → results pending in this entry.

**Results (308s, 1,480 calls).** `results/vpoc_v01.json`, `results/squish_plane_v01.png`.

- **The x-axis came alive.** Margins now spread 0.45 (`hotdog_sandwich`, squishiest) to
  0.92 (`tomato_fruit`, most robust); v0 had 7/8 pinned at 1.0. The continuous
  shift-based metric fixed the dead axis.
- **F-005 — real knife-edges.** `pluto_planet` and `cereal_soup` sit low on *both* axes:
  the model is nearly unanimous on rerun (disp 0.02 / 0.07) yet a rephrasing shifts the
  answer (margin 0.58 / 0.47). The observational axis alone called these "solid"; the
  interventional axis exposed them. This is the payoff of the two-axis design, on real data.
- **F-006 — `reorder` dominates.** It's the hot-spot for water/hotdog/prime/pluto. Real
  signal *and* a lesson: the colon-fronting construction is stilted, so some of it is
  paraphrase quality, not pure model fragility. The auditable hot-spot did its job.
- **F-007 — honestly under-powered.** Every point plotted hollow (`unresolved`): at this N,
  the CIs straddle the thresholds, so quadrant labels aren't trustworthy yet. The
  framework flagged its own uncertainty instead of faking crisp quadrants. Good. The fix
  is more samples (or a continuous plane with deadzones), not a different metric.
- `margin_max_ci` earned its place: it separates *confident* fragility (hotdog/pluto/cereal,
  significant hot-spots) from *maybe-noise* (`tomato_fruit`, hot-spot `ns`, max_ci 1.00).

---

## Open questions / backlog

- **Config as a squish axis, quantified.** Re-run the same battery under ollama's
  Modelfile defaults vs the controlled config, and directly measure how much the
  dispersion moves. That turns D-003's *argument* into *data*.
- **Kernel/backend nondeterminism.** Even with identical explicit config, different
  backends/kernels diverge (the batch-invariance issue from the Thinking Machines
  piece). Probe: does `temperature 0` give bit-identical output across reruns here?
- **Multiple-comparisons on the worst-case margin.** As K grows, more chances for a
  false-significant shift. If the taxonomy scales past ~10 types, use 99% CIs or
  Bonferroni, or report a high quantile (p90) instead of the strict max.
- **Paraphrase validation.** Worst-case + a sloppy paraphrase = false squish. Consider
  a cheap check that paraphrases are genuinely meaning-preserving (embedding distance,
  or a second model's agreement) before trusting a hot-spot.
- **The model headline.** Not yet built — a decomposable squish factor across the whole
  battery (central tendency + worst-offenders + per-type breakdown), per Kate's
  "headline number + factors + hot-spots" design.
