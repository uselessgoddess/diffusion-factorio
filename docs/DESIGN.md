# Design: masked discrete diffusion for factory layouts

## Why diffusion, and why the *masked* (absorbing-state) kind

The task asks for a **Discrete Spatial Diffusion Model**, inspired by
diffusion-for-text (DiffusionGemma-style). A factory layout is a 2D field of
*discrete* categories, so continuous Gaussian diffusion is a poor fit. The
family that works on discrete data is **absorbing-state / masked discrete
diffusion** (the lineage behind MaskGIT, D3PM-absorbing, MDLM, and text
diffusion models): instead of adding Gaussian noise, the forward process
progressively replaces tokens with a special **`MASK`** ("absorbing") token, and
the model learns to reverse it.

This choice is not incidental — it matches the *actual product requirement*:

> the model should complete a **partial** factory (place the missing entities
> given some fixed context).

That is exactly **inpainting**: the observed cells are conditioning that are
*never masked*, and the missing cells are `MASK` tokens the model fills in. The
same trained model does unconditional generation (everything masked) and
conditional completion (only some cells masked) with no change — the reference's
autoregressive+RL stack cannot do this as cleanly.

## Forward (noising) process

Continuous time `t ∈ (0, 1]`, linear schedule. Each *generative* cell is
independently replaced by `MASK` with probability `t`:

- `t = 1`: the whole non-observed grid is masked. A configurable 25% of
  training examples use this exact state because sampling always starts there.
- `t → 0`: nothing is masked.
- **Observed cells are never masked** — that is what makes the process
  conditional. (`diffusion::apply_masking`.)

Each of the 4 channels has its own `MASK` id (`VOCAB[c]`, one past the real
classes), so the model can tell "this channel is unknown" per channel.

## Reverse (denoising) process — the network

`f_θ(x_t, obstacle, t) → logits over x_0` for every channel of every cell
(`src/model.rs`, `Denoiser`). Architecture:

- **Per-channel embeddings** (`VOCAB[c] + 1` rows, the `+1` is `MASK`). Categories
  are embedded, never fed as ordinals.
- A **conv stem** consuming the concatenated channel embeddings **plus the
  obstacle conditioning plane**.
- A **residual conv tower**. Each block adds two things a plain conv can't get:
  - a **global-context vector** (concatenated mean+max pool over space → linear
    → broadcast),
    which fixes the reference's receptive-field bottleneck — routing to a distant
    sink is a grid-global decision; and
  - a **time-conditioning vector** (sinusoidal `t` → MLP, FiLM-style add), so the
    network knows how noisy its input is.
- **One 1×1 conv head per channel**, predicting *real* classes only (never
  `MASK`). All channels are produced from a shared trunk, so they are denoised
  **jointly** and stay mutually consistent.

## Objective

For every masked cell we take the cross-entropy of the predicted `x_0`
distribution against the true category, summed over the 4 coupled channels
(`diffusion::loss`). Two options:

- **Mean cross-entropy over masked cells** (MaskGIT-style, the robust default) —
  low variance, good for short runs.
- **MDLM continuous-time ELBO weight `1/t`** (`--elbo`) — a principled negative
  log-likelihood bound; higher variance, enable once training is stable.

### Fighting empty-cell dominance

The entity channel is ~95% `Empty`, so an unweighted loss is minimized by
predicting empty everywhere. We counter this directly:

- **Structure-weighted loss**: masked cells whose target entity is non-empty are
  up-weighted by `structure_weight` (default 8×). (`DiffusionConfig`.)
- **Placement-recall metric**: entity accuracy *restricted to non-empty target
  cells* — the honest "is it learning to build?" number, reported every step.
- **Assembler and recipe metrics**: anchor recall and item accuracy restricted
  to assembler targets, so belts and `Item::None` cannot hide a machine failure.

This is the most important design decision for *actually learning* rather than
collapsing to the trivial empty solution. See `docs/ROADMAP.md`.

## Inference / sampling

Confidence-based iterative decoding (MaskGIT-style,
`sample::reconstruct`): start from the partial factory (observed cells fixed,
rest `MASK`); over `steps` rounds predict `x_0` everywhere, then commit the most
confident still-masked cells (cosine reveal schedule) and re-predict the rest.
Observed cells are held fixed the whole way — exactly conditional inpainting.
Greedy (argmax) decoding is deterministic and used for reconstruction
validation; a temperature knob enables stochastic sampling for diversity.

The loop runs host-side over small validation batches so the reveal/confidence
logic is fully inspectable — the issue explicitly asked for inference we can
*always validate*.

## Representation summary (`src/world.rs`)

| Channel | Vocab | Classes |
|---|---|---|
| Entity | 8 | Empty, Source, Sink, TransportBelt, UndergroundBelt, Splitter, Inserter, Assembler |
| Direction | 5 | None, N, E, S, W |
| Item | 6 | None, IronPlate, CopperPlate, IronGear, CopperCable, GreenCircuit |
| Misc | 3 | None, UndergroundDown, UndergroundUp |

Obstacles (buildable footprint) are a **separate conditioning plane**, never a
generative channel — avoiding the reference's footprint data leak.

## Backends

`burn` with two backends selected by feature (`src/backend.rs`):

- **ndarray (CPU)** — always on; used by unit tests and CI smoke training.
- **wgpu (GPU)** — `--features wgpu`; the path for real training on the user's
  16 GB AMD rx 9070 xt. The training loop and model are backend-generic, so the
  only difference is the type alias in the binary.
