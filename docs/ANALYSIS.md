# Analysis of `beyarkay/factorion` — what to borrow, what to reject

This project is a **from-scratch Rust** re-imagining of the idea behind
[`beyarkay/factorion`](https://github.com/beyarkay/factorion): a neural model that
lays out working Factorio factories. Before writing any code we read the
reference in detail. This document records that analysis — the good ideas we
adopted, and the "childhood diseases" (детские болезни) and architectural
limits we deliberately avoided.

The reference is a hybrid: a Python model/training stack (`factorion.py`,
`sft.py`, `ppo.py`, `training_config.py`) plus a Rust simulation crate
(`factorion_rs/` — `world`, `types`, `entities`, `throughput`, `factory_gen`,
`graph`, `textual`, `render`). Our task explicitly requires **Rust only** and the
**`burn`** framework, so the Python half is reimplemented, not reused.

## What the reference gets right (borrowed)

1. **Grid-of-categoricals representation.** A factory is a fixed-size 2D grid;
   each cell is described by several *categorical* channels (entity kind,
   direction, item/recipe, and a "misc" tag). Category ids are embedded, never
   fed as raw ordinals, so no false ordering is imposed on unordered classes.
   → We keep this exactly (`src/world.rs`), trimmed to 4 channels with small
   vocabularies (`VOCAB = [8, 5, 6, 3]`).

2. **Multi-head, jointly-consistent cells.** A legal cell has mutually
   consistent channels (an underground belt *must* carry an up/down tag; a belt
   *must* face somewhere). The reference predicts channels with separate heads.
   → We denoise **all channels jointly** and score channel consistency as a
   validation metric (`Cell::is_consistent`).

3. **Procedural "lessons" as training data.** Rather than scrape blueprints, the
   reference *generates* known-correct factories by construction (curriculum of
   layout patterns) and verifies them with a simulator. This gives unlimited,
   labelled, difficulty-graded data.
   → We adopt the same strategy (`src/factory_gen.rs`): four orthogonal lesson
   kinds, each built by construction and verified functional before use.

4. **A simulator-grounded reward/metric.** The reference computes *normalized
   throughput* by building a lane-aware flow graph and propagating flow
   (`throughput.rs`, `graph.rs`). This is the signal that tells you the model
   learned something *functional*, not just token-accurate.
   → We keep the spirit with a cheaper but meaningful check — does the item flow
   from source to sink along the placed belts? (`src/sim.rs`,
   `item_reaches_sink`). Full lane-aware throughput is a roadmap item.

5. **Textual / renderable factories.** The reference has a YAML text form and a
   renderer, which makes outputs human-checkable.
   → We keep a lightweight single-glyph ASCII view (`src/textual.rs`) so every
   inference result can be eyeballed.

## What we reject or fix (the "childhood diseases")

1. **The `FOOTPRINT` data leak.** In the reference, the buildable-footprint
   channel at one point effectively encoded *where the answer went* (only the
   correct placement cells were marked buildable), letting the model cheat.
   → We make obstacles a **separate conditioning input** that never encodes where
   entities should go (`Grid::obstacle`), and it is *not* a generative channel.
   A blank grid has no obstacles by default.

2. **Receptive-field bottleneck.** A purely local conv/window model cannot answer
   grid-global questions ("which way is the far-off sink?"), which caps layout
   quality. This is a real architectural limit the reference bumped into.
   → Every residual block injects a **global-context vector** (mean-pool → linear
   → broadcast), so global routing information is available everywhere
   (`src/model.rs`, `ResBlock`). See `docs/DESIGN.md`.

3. **Autoregressive/RL sequ-of-tokens framing.** The reference leans on
   supervised fine-tuning then PPO over a token sequence — heavy, sample-hungry,
   and awkward for *conditional* placement (fill in the missing entities given a
   partial factory).
   → We reframe generation as **masked discrete diffusion / inpainting**: observed
   cells are conditioning, missing cells are MASK tokens the model fills. This is
   naturally conditional, parallel to decode, and needs no RL to get started.
   (Full rationale in `docs/DESIGN.md`.)

4. **Class imbalance (empty-cell / EOT dominance).** Most of a grid is empty, so
   an unweighted objective is trivially minimized by predicting "empty"
   everywhere — the model *looks* accurate (~95%) while learning nothing. The
   reference had to fight this imbalance.
   → We surface it with an honest metric (**placement recall**: entity accuracy
   restricted to non-empty target cells) and counter it with a
   **structure-weighted loss** (`DiffusionConfig::structure_weight`). See
   `docs/ROADMAP.md` — this is the single most important bottleneck to watch.

5. **Lesson-sampling starvation.** If some lessons rarely generate (rejection
   sampling fails), the curriculum silently starves.
   → Our generator retries kinds/seeds and the training loop draws only from
   **feasible** kinds for the grid size (`train::feasible_kinds`), so no lesson is
   silently dropped.

## Net design stance

Keep the reference's strongest ideas — categorical grid, jointly-consistent
cells, procedurally-generated verified lessons, simulator-grounded metrics — and
swap its generation paradigm for **masked discrete diffusion**, while fixing the
data leak, the receptive-field limit, and the empty-cell imbalance up front so
they don't resurface as the same "childhood diseases".

## Follow-up analysis after the first GPU convergence run

The reported 5,000-step run reaches `exact=0.988`, `functional=0.988`, and
`consistent=0.996` over 256 procedural reconstructions. This is strong evidence
that the model escaped empty collapse and learned the current four-family
curriculum. It is not yet evidence that arbitrary Factorio factories are solved:
train and evaluation still share procedural lesson families, and the binary
reachability metric cannot distinguish low from high throughput.

The current Factorion repository has also moved beyond the early architecture
review. Its most useful newer patterns are held-out/per-task validation,
per-head losses, visual prediction diagnostics, blueprint/mod/RCON integration,
graded throughput rollouts, SFT-to-PPO initialization, a fast Rust throughput
engine, and a real-Factorio parity harness.

This project now borrows the parts that are prerequisites for trustworthy work:

- frozen, balanced validation independent of the advancing training stream;
- durable per-step and per-head telemetry with per-lesson splits;
- confidence, entropy, error, and reveal-time spatial diagnostics;
- a real Factorio 2.x blueprint-string round trip with visible source/sink
  markers.

The remaining order matters. First make multi-tile footprints honest and verify
a graded throughput simulator against real Factorio. Then use the diffusion
model's natural strength—parallel candidate generation—for best-of-N search and
elite replay. PPO-style or reward-weighted diffusion fine-tuning only becomes a
sound optimization target after parity prevents reward hacking.

## Re-reading the reference at `fdb723a` (2026-07)

A closer read against their current `main`, prompted by the 5,000-step run.
Full write-up in [`TRAINING_ANALYSIS.md`](TRAINING_ANALYSIS.md); the parts that
change our plan:

**Their docs disagree with their code.** `CLAUDE.md` is the only file that
matches what the code does; `README.md` and `docs/EXPERIMENTS.md` are ~8 months
stale and use a different normalization. Do not cite the 0.58–0.60 figure from
`EXPERIMENTS.md`.

**Where they actually are.** Canonical SFT base `j0s5y2mc` scores
`val/thput_eot ≈ 0.11` — a greedy rollout that blanks the *whole* grid and
rebuilds from empty. Per-lesson: `MOVE_ONE_ITEM ≈ 0.38`, **assembler lessons
≈ 0**. With PPO, 45M samples and a full throughput engine, they cannot yet build
a working assembler factory from scratch. That is the real bar, and it is lower
than their README implies.

**Newly worth borrowing:**

- **Power-mean throughput** `((1/N)·Σ achievedᵢ^p)^(1/p)` at `p=0.5`
  (`throughput.rs:7-17,39-45`) — punishes starving any one sink, which a plain
  average does not. This is the single unlock: it makes the objective graded.
- **1×1-conv tile head → softmax over the flat board** (`ppo.py:1243`). Their
  PR #16: 2.6M → **520 parameters (~5000×)**, no throughput loss, **+76.4% SPS**
  (p=4.7e-09). Nearly free.
- **Per-tile conditioned attribute heads** `P(tile)·P(attrs|tile)` — directly
  relevant to our joint-consistency problem.
- **`thput_eot` metric discipline** — blank everything, rebuild from empty. We
  have now adopted this as the `SCRATCH` validation pass.
- **Lesson balancing on emitted (state, action) pairs**, not factories
  (`sft.py:263-266`), because big factories emit ~10× more pairs. Relevant once
  our lessons differ in size.

**Newly rejected:**

- **Their assembler throughput is wrong.** `entities.rs:426-451` never reads
  `crafting_time` or `crafting_speed` and caps at `min_ratio ≤ 1.0` — a
  pass-through ratio, not a 0.5-craft/s machine. Port the flow graph, not this.
- **`thput_normed` needs the scripted reference solution** to normalize, so it
  only exists for lessons they can already solve by construction — a live
  `FIXME(#161)` at `ppo.py:654-665`. Our score should be absolute.
- **Argmax-only inference** can loop on a tile; our confidence-ordered reveal is
  better, and best-of-N better still.
- **Potential-based reward shaping** — they tried it and rejected it (PR #18:
  −2.8% throughput at p=0.560, i.e. not significant, for −18.3% SPS). If we ever
  reach RL, reward stays terminal-only.

**Already ours:** global mean+max pooled context (`ppo.py:1245-1258`, their #290)
is in `src/model.rs`.

**Convergent, which is reassuring:** their dataset is 100% procedural — nothing
scraped, no human blueprints — streamed online (`StreamingDemoDataset`,
`sft.py:323-350`, `num_samples = 45_000_000`, `epochs = 1`), validated by the
throughput engine itself, with train/val disjoint by seed arithmetic
(`sft.py:932-946`). Same principles we arrived at independently. The difference
is **scale**: 45M samples versus our 160k.

One caution against a plan we might otherwise adopt: they **explicitly reverted
a held-out-recipe split** (#272) on the grounds that "memorising recipes is
desired behaviour".
