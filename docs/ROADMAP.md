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

## The metrics that prove learning (watch these)

Per training step we log, and validation aggregates:

- **`place` (placement recall)** — entity accuracy on masked **non-empty** cells.
  This is the honest signal; unlike raw entity accuracy it is *not* inflated by
  the empty-cell majority. If this is near 0 while loss drops, the model has
  collapsed to "predict empty".
- **`functional`** — fraction of *reconstructed* factories where the item still
  reaches a sink (simulator-grounded). The number that actually matters.
- **`exact`** — fraction reconstructed exactly on masked cells.
- **`consistent`** — fraction of reconstructions that are well-formed cells.
- Per-channel accuracy `[entity, dir, item, misc]`.

## Bottlenecks, ranked

### 1. Empty-cell dominance (the big one)
~95% of cells are `Empty`. An unweighted loss collapses to predicting empty
everywhere — high apparent accuracy, zero functional factories. This was
observed directly in early smoke runs (`functional` fell to 0 as the model
collapsed).
**Mitigation in place:** `structure_weight` (up-weight non-empty targets) +
placement-recall metric to detect collapse.
**Next:** tune `structure_weight`; try focal loss; try masking the *removable*
cells preferentially during training so the belts are always the learning target.

### 2. Simulator fidelity
`item_reaches_sink` is a reachability check, not true lane-aware throughput. It
can accept layouts Factorio would consider imperfect (e.g. it does not model belt
capacity, sideloading, or splitter balancing).
**Next:** port the reference's lane-aware flow graph (`graph.rs` + `throughput.rs`)
to give a graded *normalized-throughput* metric and, later, an RL reward.

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

### 5. Compute path (CPU vs GPU)
CI trains on ndarray/CPU (slow, smoke-only, ~1 s/step). Real training needs the
wgpu backend.
**Next:** run `--features wgpu` on the 16 GB rx 9070 xt; profile step time and
batch size; confirm parity of results between backends.

## Concrete next steps (in order)

1. **Convergence study on GPU** — run `train --features wgpu` for ≥50k steps;
   confirm `place` → high and `functional` → high on held-out lessons. Record a
   curve in this file.
2. **Tune the imbalance knobs** — sweep `structure_weight`, add focal loss,
   compare mean-CE vs `--elbo`.
3. **Lane-aware throughput** — port `graph.rs`/`throughput.rs`; switch the
   headline metric from binary "reaches sink" to graded throughput.
4. **Richer curriculum** — multi-tile assemblers, buses, branches; held-out
   kinds for generalization.
5. **Multi-scale denoiser** — U-Net down/up path; measure large-grid gains.
6. **RL fine-tuning (optional)** — once throughput is graded, add a PPO-style
   pass on top of the diffusion prior (mirrors the reference's SFT→PPO, but with a
   much stronger, conditional starting point).
7. **Blueprint export** — map the grid back to a Factorio blueprint string for
   real in-game validation (the ultimate "is it SOTA / usable" check).

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
```
