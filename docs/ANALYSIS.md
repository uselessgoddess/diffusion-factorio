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
