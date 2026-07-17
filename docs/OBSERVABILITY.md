# Training observability

Each `train` run produces two artifacts by default:

- `runs/training-metrics.jsonl` is flushed after every completed optimizer step.
  It survives an interrupted long GPU run and is easy to load from Python,
  DuckDB, jq, or a spreadsheet converter.
- `runs/training-report.html` is a dependency-free report generated at normal
  completion. It embeds the metrics, so it can be opened offline or archived
  next to the checkpoint.

Use `--metrics-out` and `--report-out` to group artifacts per experiment:

```bash
cargo run --release --features wgpu --bin train -- \
  --steps 5000 --seed 7 \
  --out runs/seed-7/denoiser \
  --metrics-out runs/seed-7/metrics.jsonl \
  --report-out runs/seed-7/report.html
```

## Reading the curves

`loss` is the optimized, structure-weighted objective. Compare it with each
channel's unweighted NLL to see whether one head is dominating.
`placement_recall` is entity accuracy only on masked non-empty targets; it
exposes all-empty collapse. Raw entity accuracy alone is not trustworthy because
most cells are empty.

`assembler_recall` narrows that signal to assembler anchors, and
`recipe_accuracy` scores the item channel only at those anchors. Read both for a
factory task: aggregate placement is mostly belts and aggregate item accuracy is
mostly `Item::None`, so neither can prove the model learned machines.

`functional_rate` asks the internal simulator whether reconstructed layouts
still connect a source to a sink. `exact_rate` requires every masked category to
match the procedural solution, while `consistent_rate` only requires legal
channel combinations. A functional result can legitimately be non-exact.

Validation is frozen at run start from an independent deterministic seed and is
balanced round-robin over every feasible lesson family. Consequently, changes
between validation points are model changes rather than a different random
batch. `val_by_lesson` in JSONL separates routing, chaos, assembler, and
underground performance so an aggregate cannot hide a weak lesson.

The 64-sample default validation still has sampling uncertainty. At a measured
rate near 0.95 its standard error is about 0.027; compare seeds and increase
`--val-batch` for checkpoint decisions.

## Spatial diagnostics

`sample` writes `sample-report.html` (override with `--report`). For every shown
example it contains:

- masked input, prediction, and ground truth;
- confidence: mean selected-class probability across the four categorical heads;
- entropy: mean per-head entropy normalized to `0..1`;
- error: cells that differ from the target, excluding observed conditioning;
- reveal round: when confidence-based decoding committed each cell.

Confidence and entropy are captured at commitment time. Computing them from a
final pass would let the model see its own completed tokens and exaggerate
certainty.

The default command measures partial inpainting. Use `--scratch` to leave only
the source/sink task anchors, and `--lesson ASSEMBLER_CHAOS` (or another report
name) to isolate one family:

```bash
cargo run --release --bin sample -- --ckpt checkpoints/denoiser \
  --scratch --lesson ASSEMBLER_CHAOS --size 13 --height 9 --eval 128
```

Scratch validation also reports assembler/recipe recall and inserter/belt
entity-plus-direction recall. These component metrics distinguish a missing
machine from a disconnected route; aggregate cell accuracy cannot, because most
of the canvas is correctly empty in both cases.

## Important parameters

- `--structure-weight`: multiplier for non-empty target cells. Too low permits
  empty collapse; too high over-places structure.
- `--warmup`, `--lr`: linear warmup to the peak learning rate, followed by cosine
  decay.
- `--t-min`: lower bound on diffusion time, which bounds `1/t` variance when
  `--elbo` is enabled.
- `--scratch-probability`: fraction trained at exactly `t=1`, the fully masked
  state from which scratch sampling begins (default 0.25).
- `--sample-steps`: confidence-based reveal rounds. More rounds provide more
  opportunities to revise uncertain regions but cost more validation time.
- `--hidden`, `--blocks`: convolutional width and depth; both raise memory and
  compute cost.
- `--val-batch`: size of the fixed held-out corpus. Increase it to reduce
  validation noise before comparing close checkpoints.
- `--no-assembler-open`: local/GPU control arm that removes only the
  obstacle-free assembler bridge. Keep the same seed and all other flags when
  comparing it with the production curriculum.
