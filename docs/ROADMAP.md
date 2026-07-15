# Roadmap & bottlenecks

The issue asks specifically that **future training paths and bottlenecks be
visible and clear**, and that we always have **metrics + validatable inference**
proving the model is *really learning*. This document is that map: what works
now, what the known bottlenecks are (ranked), and the concrete next steps.

## Status: what works today

- **World model** (`src/world.rs`) — 4-channel categorical grid, consistency
  rules, obstacles as separate conditioning. ✅ unit-tested.
- **Simulator** (`src/sim.rs`) — `item_reaches_sink` functional check for belts,
  undergrounds, inserters, assemblers. ✅ unit-tested.
- **Lesson generator** (`src/factory_gen.rs`) — 4 lesson kinds, built by
  construction and verified functional; blanking into (partial, solution) pairs.
  ✅ unit-tested.
- **Masked diffusion core** (`src/diffusion.rs`) — forward masking + joint,
  structure-weighted CE loss, MDLM ELBO option. ✅ unit-tested.
- **Denoiser** (`src/model.rs`) — per-channel embeddings, conv tower with
  global-context + time injection, per-channel heads. ✅ shape-tested.
- **Training loop** (`src/train.rs`) — AdamW, warmup+cosine LR, grad clipping,
  streaming data, periodic functional validation. ✅ smoke-tested (loss drops).
- **Inference** (`src/sample.rs`) — iterative confidence-based inpainting;
  `sample` binary reconstructs blanked factories and reports functional validity.
- **Binaries** — `gen_data`, `train`, `sample`. ✅ build + run.
- **CI** — build, clippy, fmt, unit tests, and a tiny CPU training smoke test.
- **Frozen evaluation + observability** — deterministic held-out corpus,
  per-step JSONL, offline curve report, per-lesson metrics, and spatial
  confidence/entropy/error/reveal heatmaps.
- **Factorio export** — vanilla entity/direction/recipe mapping and Factorio 2.x
  compressed blueprint strings, including visible source/sink markers.

## The metrics that prove learning (watch these)

Per training step we log, and validation aggregates:

- **`place` (placement recall)** — entity accuracy on masked **non-empty** cells.
  This is the honest signal; unlike raw entity accuracy it is *not* inflated by
  the empty-cell majority. If this is near 0 while loss drops, the model has
  collapsed to "predict empty".
- **`functional`** — fraction of *reconstructed* factories where the **right
  item** still reaches a sink (simulator-grounded). The number that actually
  matters.
- **`exact`** — fraction reconstructed exactly on masked cells.
- **`consistent`** — fraction of reconstructions that are well-formed cells.
- Per-channel accuracy `[entity, dir, item, misc]`.

Each is reported in **two modes**, and the difference is the point:

- **inpaint** — fill the gaps in a given scaffold. Historical metric, kept for
  comparability. Easy: 2–7 masked cells of 121.
- **`SCRATCH`** — only the source and sink are visible (~119 of 121 masked), so
  the model must decide *what to build and where*. **Read `functional` here, not
  `exact`**: many layouts deliver the item, so `exact` only rewards
  rediscovering the generator's own BFS answer. This is the first metric in the
  project where the two genuinely come apart — under inpainting the data forces
  them to agree.

## Bottlenecks, ranked

### 0. The task is too small, and the eval could not tell us (the real one)
Measured, not guessed — see [`docs/TRAINING_ANALYSIS.md`](TRAINING_ANALYSIS.md)
and `cargo run --release --example task_space`. The 5,000-step GPU run reached
`exact=1.000 functional=1.000`, and that number is close to meaningless:

- **`assembler_line` asks the model to fill 2.0 cells out of 121** (1.7% of the
  grid), both always `Inserter, East`, from **231 distinct templates** seen ~173×
  each. `underground_cross`: 110 templates, ~364× each. That is memorization
  scale. (`move_one_item` and `..._chaos` are the honest half — ~42k and 200k+
  distinct tasks, each seen ~once, so 1.000 there is real generalization.)
- **`ambiguous tasks: 0` everywhere.** Each conditioning has exactly one valid
  answer, so `functional == exact` is a property of the *data*, and a 30-line BFS
  beats the model at the task as posed.
- **`exact=1.000` came from n=64.** For an all-successes run the 95% lower bound
  is `0.05^(1/n)`: 64/64 proves only >95.4%, and per-lesson 16/16 only >82.9%.
  The fresh (held-out) training batches put the real entity error at **0.19%**
  and show `place < 1.0` on **16.8% of batches** — a tail the frozen set is too
  small to contain.

**Mitigations in place:** from-scratch validation (`Sample::blank_to_scaffold`)
masks everything but the source/sink, so the model must *design*, not inpaint;
`val_batch` default 64 → 512; `functional` is now item-aware.
**Next:** a curriculum where tasks admit *many* valid answers, which is the
precondition for the model to beat BFS at all.

### 1. Empty-cell dominance (the big one)
~95% of cells are `Empty`. An unweighted loss collapses to predicting empty
everywhere — high apparent accuracy, zero functional factories. This was
observed directly in early smoke runs (`functional` fell to 0 as the model
collapsed).
**Mitigation in place:** `structure_weight` (up-weight non-empty targets) +
placement-recall metric to detect collapse.

**Empirical check (CI smoke-train, 200 CPU steps, `structure_weight=8`):**
```
AGGREGATE: n=128 | exact=0.008 functional=0.258 (orig_fn=128)
           consistent=0.805 | acc[entity=0.044 dir=0.110 item=1.000 misc=0.892]
```
`functional=0.258` (vs `0.0` at collapse) after only 200 steps confirms the fix
escapes the empty attractor — the model builds real, partly-working structure.
The low `entity` accuracy is the *opposite* symptom: at `structure_weight=8` the
model now **over-places** structure (predicts belts on cells that should be
empty), so it loses the empty-cell accuracy it used to farm for free. That is a
much healthier failure than collapse and is a tuning target, not a wall.
**Next:** sweep `structure_weight` down toward the true non-empty ratio; try
focal loss; try masking the *removable* cells preferentially during training so
the belts are always the learning target.

### 2. Simulator fidelity — the metric cannot rank two working factories
`item_reaches_sink` is a *binary* reachability check, not lane-aware throughput:
it does not model belt capacity, sideloading, or splitter balancing, so it cannot
say which of two working layouts is better.

This is the blocker for almost everything downstream. Best-of-N has nothing to
sort by; RL has no gradient to climb (the binary reward is already saturated at
1.0). **Graded throughput is the unlock, and it should come before RL.**

Fixed since the last run: `item_reaches_sink` was also **item-blind**, scoring
"belt raw plate straight into a gear sink" as functional — i.e. rewarding
*skipping* the assembler. It now carries the item through the BFS and applies
recipes (the reference guards the same hole at `throughput.rs:205-226`).
**Next:** port the reference's lane-aware flow graph (`graph.rs` +
`throughput.rs`), scoring `((1/N)·Σ achievedᵢ^p)^(1/p)` at `p=0.5` so starving
any one sink is punished. Note their assembler model is **wrong** — it never
reads `crafting_time`/`crafting_speed` (`entities.rs:426-451`) — so port the
structure, not that part.

### 3. Receptive field / global routing
Addressed architecturally via the global-context vector, but for large grids a
single mean-pool may be too coarse.
**Next:** multi-scale U-Net (down/up sampling) or axial/attention blocks; measure
whether functional-rate scales with grid size.

### 4. Curriculum breadth & realism
Four hand-built lessons exercise every channel but are small and templated. Real
Factorio layouts are richer (3×3 assemblers, multi-input recipes, buses).
**Next:** grow the lesson set (true multi-tile buildings, branching buses),
weight the curriculum by difficulty, and add held-out lesson kinds to measure
generalization.

### 5. Compute path — the GPU is idle, and the schedule wastes 40% of the run
Not a wall, but free money. Profiled from the 5,000-step run's report:

```
total elapsed 140.5 s (2.3 min) | median train step 27.76 ms
data generation 13,811-22,787 gen/s (~1.5-2.3 ms per batch of 32) = ~7% of step
validation 2.1 s total = 1.5% of the run
```

- **Data generation is not the bottleneck** (~7%), so the streaming design is
  fine and batch 32 simply underutilizes the GPU. Raise it.
- **Metrics saturate at step ~3,000 but cosine decay runs to 5,000**, ending at
  `lr 3.08e-11`. The last ~2,000 steps did not move the weights: **~40% of the
  run was spent not learning.** The gradual curve is the LR tail, not difficulty.

**Next:** raise batch size until the GPU saturates; match schedule length to
where learning actually stops; confirm backend parity (ndarray vs wgpu).

## Concrete next steps (in order)

Reordered after the 5,000-step run —
see [`docs/TRAINING_ANALYSIS.md`](TRAINING_ANALYSIS.md) for the evidence.

1. **From-scratch evaluation** ✅ *(this branch)* — mask everything but the
   source/sink so the model designs instead of inpainting. Establishes the real
   baseline. Expect `exact` to collapse and `functional` to become the number
   that matters; the reference scores ~0.11 on the equivalent metric.
2. **Lane-aware graded throughput** — port `graph.rs`/`throughput.rs` (power mean
   at `p=0.5`). Everything below is blocked on this: it is what makes one working
   factory rankable against another.
3. **Best-of-N sampling, verified by the simulator** — the highest-leverage step
   for *usable output*. Sample N layouts, simulate each, keep the best. The
   sampler is already stochastic and conditional, so this needs **no retraining**
   and converts "right 99.8% of the time" into "the exported blueprint works".
   Pipeline: generate → verify → best-of-N → export.
4. **Richer curriculum that admits many answers** — multi-source/multi-sink,
   several recipes, tighter obstacle budgets, true 3×3 assemblers and 2×1
   splitters. Until a task has more than one valid solution, the model cannot
   demonstrate design and BFS remains the better tool.
5. **Tune the imbalance knobs** — sweep `structure_weight`, add focal loss,
   compare mean-CE vs `--elbo`.
6. **Cheap architecture wins from the reference** — 1×1-conv tile head → softmax
   over the flat board (their PR #16: 2.6M → **520 params**, no throughput loss,
   +76.4% SPS); per-tile conditioned attribute heads `P(tile)·P(attrs|tile)`.
7. **Factorio parity** — RCON harness (1800 warmup / 3600 measure ticks at 32×)
   to prove the simulator is not lying. Not CI-able; needs a licensed install.
8. **RL/self-improvement (optional, last)** — only once throughput is graded and
   parity-checked. Today the reward is binary and already saturated, so **there
   is nothing for a policy gradient to climb**. Note the reference *tried*
   potential-based shaping and rejected it (−2.8% thput at p=0.560, −18.3% SPS);
   reward stays terminal-only.

## How to reproduce

```bash
# Inspect the data
cargo run --bin gen_data -- --size 11 --count 1

# Train (CPU smoke)
cargo run --release --bin train -- --steps 2000 --val-every 200

# Train (GPU, real)
cargo run --release --features wgpu --bin train -- --steps 50000 --out checkpoints/denoiser

# Validate: blank known factories and reconstruct
cargo run --release --bin sample -- --ckpt checkpoints/denoiser --show 4 --eval 256

# Inspect heatmaps and import the first reconstruction in Factorio
cargo run --release --bin sample -- --ckpt checkpoints/denoiser \
  --blueprint-out generated-blueprint.txt
```
