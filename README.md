# diffusion-factorio

A from-scratch **Rust** implementation of a **discrete spatial diffusion model**
that generates working [Factorio](https://factorio.com) factory layouts. Built on
the [`burn`](https://burn.dev) deep-learning framework (v0.21), training on CPU
(ndarray) for CI and on the **GPU via wgpu** for real runs.

It is a re-imagining of the idea behind
[`beyarkay/factorion`](https://github.com/beyarkay/factorion) — keeping its
strongest ideas (categorical grid, jointly-consistent cells, procedurally
generated *verified* lessons, simulator-grounded metrics) and swapping its
autoregressive + RL generation stack for **masked discrete diffusion**, which is
naturally *conditional* (complete a partial factory) and needs no RL to start.
See [`docs/ANALYSIS.md`](docs/ANALYSIS.md) for the full borrow-vs-reject analysis.

## Why masked diffusion

A factory layout is a 2D field of *discrete* categories, so continuous Gaussian
diffusion is a poor fit. We use **absorbing-state / masked discrete diffusion**
(the family behind MaskGIT, D3PM-absorbing, MDLM, and DiffusionGemma-style text
diffusion): the forward process replaces cells with a `MASK` token, and the
denoiser learns to reverse it.

The payoff is that **completing a partial factory is just inpainting** — the
observed cells are conditioning that is never masked; the missing cells are
`MASK` tokens the model fills in. The same model does unconditional generation
(everything masked) and conditional completion (some cells masked) with no
change. Full design in [`docs/DESIGN.md`](docs/DESIGN.md).

## Representation

Each grid cell is 4 categorical channels (`src/world.rs`):

| Channel | Vocab | Classes |
|---|---|---|
| Entity | 8 | Empty, Source, Sink, TransportBelt, UndergroundBelt, Splitter, Inserter, Assembler |
| Direction | 5 | None, N, E, S, W |
| Item | 6 | None, IronPlate, CopperPlate, IronGear, CopperCable, GreenCircuit |
| Misc | 3 | None, UndergroundDown, UndergroundUp |

Buildable obstacles are a **separate conditioning plane**, never a generative
channel — avoiding the reference's footprint data leak.

## Is it really learning? (metrics)

The issue's central requirement: it must always be *clear the model is really
learning*. Every validation step blanks a set of known-good factories,
reconstructs them, and reports:

- **`place` — placement recall**: entity accuracy on masked *non-empty* cells.
  The honest signal, immune to the empty-cell majority. If loss drops but `place`
  stays near 0, the model has collapsed to "predict empty".
- **`functional`**: fraction of *reconstructed* factories where the item still
  reaches a sink, checked by the simulator (`src/sim.rs`). The number that matters.
- **`exact`**, **`consistent`**, and per-channel accuracy `[E, D, I, M]`.

The single biggest bottleneck — **empty-cell dominance** (~95% of cells are
empty) — is countered with a **structure-weighted loss** and surfaced by the
placement-recall metric. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the ranked
bottleneck list and next steps.

## Quickstart

```bash
# Inspect the procedurally-generated lessons (solution + masked view)
cargo run --bin gen_data -- --size 11 --count 1

# Train a small model on CPU and save a checkpoint (smoke-scale)
cargo run --release --bin train -- --steps 2000 --val-every 200 --out checkpoints/denoiser

# Real training on the GPU (wgpu / AMD rx 9070 xt)
cargo run --release --features wgpu --bin train -- --steps 50000 --out checkpoints/denoiser

# Validatable inference: blank known factories, reconstruct, and score
cargo run --release --bin sample -- --ckpt checkpoints/denoiser --show 4 --eval 256
```

## Crate layout

| Module | Purpose |
|---|---|
| `world.rs` | Grid, cells, channel vocab, consistency rules |
| `sim.rs` | Item-flow simulator (`item_reaches_sink`) — the functional metric |
| `factory_gen.rs` | Procedural lesson curriculum (verified functional) |
| `data.rs` | Grids → `burn` tensors (`GridBatch`) |
| `diffusion.rs` | Masking forward process + structure-weighted loss |
| `model.rs` | The `Denoiser` (conv tower + global-context + time conditioning) |
| `train.rs` | Explicit AdamW loop, LR schedule, periodic functional validation |
| `sample.rs` | Confidence-based iterative decoding (inpainting) |
| `metrics.rs` | `reconstruction_report` — the "is it learning?" scorer |
| `persist.rs` | Checkpoint save/load (`CompactRecorder` + JSON config) |
| `textual.rs` | ASCII rendering so every output is eyeballable |
| `bin/` | `gen_data`, `train`, `sample` CLIs |

## Backends

`burn` with two backends selected by Cargo feature (`src/backend.rs`):

- **ndarray (CPU)** — default; used by unit tests and CI smoke training.
- **wgpu (GPU)** — `--features wgpu`; the real-training path. The loop and model
  are backend-generic, so only the type alias in the binary differs.

## Documentation

- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) — reference analysis: what to borrow, what to reject.
- [`docs/DESIGN.md`](docs/DESIGN.md) — the masked-diffusion design in detail.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — bottlenecks (ranked) and next steps.
