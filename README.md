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
learning*. Every run builds one **frozen, seed-controlled validation corpus**;
each validation step reconstructs those same known-good factories and reports:

- **`place` — placement recall**: entity accuracy on masked *non-empty* cells.
  The honest signal, immune to the empty-cell majority. If loss drops but `place`
  stays near 0, the model has collapsed to "predict empty".
- **`asm` and `recipe`**: assembler-anchor recall and recipe accuracy only on
  assembler targets. These prevent belts and the `Item::None` majority from
  concealing a model that never learned to place or configure a machine.
- **`functional`**: fraction of *reconstructed* factories where the **right item**
  reaches a sink, checked by the simulator (`src/sim.rs`). The number that matters.
- **`exact`**, **`consistent`**, and per-channel accuracy `[E, D, I, M]`.

Each is reported in two modes, because the easy one flatters:

- **inpaint** — fill the gaps in a given scaffold. Only 2–7 of 121 cells are
  masked, so a good score here does not mean the model can design a factory.
- **`SCRATCH`** — only the source and sink stay visible (~119 of 121 masked): the
  model is told *"plates enter here, gears must arrive there"* and must decide
  what to build and where. **Read `functional` here, not `exact`** — many layouts
  work, so `exact` only rewards rediscovering the generator's own answer.

A warning that applies to both: with `--val-batch 64`, a perfect `1.000` is
statistically consistent with a true per-lesson rate of **83%**. The default is
512 for that reason. [`docs/TRAINING_ANALYSIS.md`](docs/TRAINING_ANALYSIS.md)
works through what the first converged GPU run does and does not prove.

The console is no longer the only record. Training flushes one structured row
per step to `runs/training-metrics.jsonl` and writes a self-contained offline
`runs/training-report.html` with loss/per-head NLL, LR, throughput, placement,
validation curves, per-lesson metrics, and an annotated parameter table.
Sampling writes `sample-report.html` with confidence, normalized entropy, error,
and reveal-round heatmaps. Confidence is captured when a cell is committed—not
after feeding the completed answer back into the model.

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

# Also export the first reconstruction to Factorio's import-string format
cargo run --release --bin sample -- --ckpt checkpoints/denoiser \
  --blueprint-out generated-blueprint.txt
```

Open `sample-report.html` to see the model's factory drawn beside the ground
truth, over the spatial uncertainty heatmaps. To look at the training data the
same way — every lesson family, drawn at Factorio's footprints, each with a
paste-ready blueprint string:

```bash
cargo run --release --example gallery   # writes gallery.html
```

Both of those show the model against tasks *someone generated*. To pose one
nobody generated — paint sources, sinks and obstacles by hand and watch the model
design the factory between them:

```bash
cargo run --release --bin serve -- --ckpt checkpoints/denoiser   # 127.0.0.1:8080
```

The server runs Best-of-N, scores every candidate through the simulator, and
shows the winner beside its runners-up with a scrubber that replays the order the
model committed cells in. Inference is CPU/ndarray whatever the checkpoint was
trained on: a viewer that needs a GPU is a viewer nobody opens. It is
single-threaded with no TLS and no auth — keep it on localhost.

To inspect the result
in Factorio, copy `generated-blueprint.txt`, open the Blueprint Library (`B`),
click **Import string**, and paste it. Abstract source/sink anchors appear as
tagged constant combinators. See [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md)
for metric interpretation and [`docs/FACTORIO_EXPORT.md`](docs/FACTORIO_EXPORT.md)
for the mapping and current simulation-parity limits.

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
| `observability.rs` | Durable JSONL, offline curve report, spatial heatmaps |
| `blueprint.rs` | Grid → Factorio 2.x JSON and compressed blueprint string |
| `persist.rs` | Checkpoint save/load (`CompactRecorder` + JSON config) |
| `textual.rs` | ASCII rendering so every output is eyeballable |
| `viewer.rs` | SVG rendering at Factorio's real footprints — the human's view |
| `throughput.rs` | Graded items/s — the metric that ranks two *working* factories |
| `best_of_n.rs` | Draw N, verify each through the simulator, keep the winner |
| `serve.rs` | Hand-painted tasks → design → simulator verdict, over HTTP |
| `bin/` | `gen_data`, `train`, `sample`, `serve` CLIs |

## Backends

`burn` with two backends selected by Cargo feature (`src/backend.rs`):

- **ndarray (CPU)** — default; used by unit tests and CI smoke training.
- **wgpu (GPU)** — `--features wgpu`; the real-training path. The loop and model
  are backend-generic, so only the type alias in the binary differs.

## Documentation

- [`docs/INFERENCE_AND_TRAINING.md`](docs/INFERENCE_AND_TRAINING.md) — how to read
  a training log without fooling yourself, what to watch locally, and where the
  project goes next. Start here.
- [`docs/RL_ANALYSIS.md`](docs/RL_ANALYSIS.md) — where the project actually is,
  and why RL is still not next.
- [`docs/TRAINING_ANALYSIS.md`](docs/TRAINING_ANALYSIS.md) — what the first
  converged GPU run does and does not prove, with re-derivable numbers.
- [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md) — why a 10,000-step run that
  reads `loss 0.1079 / acc 0.98` still builds factories that deliver nothing, and
  which of the five causes were the model's fault (none of the first four).
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) — reference analysis: what to borrow, what to reject.
- [`docs/DESIGN.md`](docs/DESIGN.md) — the masked-diffusion design in detail.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — bottlenecks (ranked) and next steps.
- [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) — how to read curves and heatmaps.
- [`docs/VIEWER.md`](docs/VIEWER.md) — seeing the factory: viewer vs blueprint vs
  mod vs RCON, what the reference does, and when each is worth building.
- [`docs/FACTORIO_EXPORT.md`](docs/FACTORIO_EXPORT.md) — in-game import and entity mapping.
