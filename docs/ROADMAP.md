# Roadmap & bottlenecks

The issue asks specifically that **future training paths and bottlenecks be
visible and clear**, and that we always have **metrics + validatable inference**
proving the model is *really learning*. This document is that map: what works
now, what the known bottlenecks are (ranked), and the concrete next steps.

Analysis of the 5,000-step GPU run, and why RL is still not the next step:
[`docs/RL_ANALYSIS.md`](RL_ANALYSIS.md).

## Status: what works today

- **World model** (`src/world.rs`) — 4-channel categorical grid, consistency
  rules, obstacles as separate conditioning. ✅ unit-tested.
- **Simulator** (`src/sim.rs`) — `item_reaches_sink` functional check for belts,
  undergrounds, inserters, assemblers. ✅ unit-tested.
- **Lesson generator** (`src/factory_gen.rs`) — 5 lesson kinds, built by
  construction and verified functional; blanking into (partial, solution) pairs.
  One of them (`ASSEMBLER_BANK`) admits **many** valid answers per task.
  ✅ unit-tested.
- **Graded throughput** (`src/throughput.rs`) — items/second per sink, folded by
  a power mean at `p=0.5`. Ranks two *working* factories against each other.
  ✅ unit-tested.
- **Best-of-N** (`src/best_of_n.rs`) — draw N candidates, keep the one the
  simulator scores highest. No retraining. ✅ unit-tested.
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
  rediscovering the generator's own BFS answer.

Two ways the metrics come apart, and they are different things. `SCRATCH` makes
`exact` *hard* — the model must rediscover one specific layout out of many that
work. `ASSEMBLER_BANK` makes `exact` **wrong**: the task has three valid answers,
so `exact` is capped below 1.0 by construction and a model that always builds
the best answer scores *worse* on it than one that guesses the generator's roll.
On ambiguous families `exact` is a diagnostic, not a target.

Since throughput landed, validation also reports:

- **`thput`** — mean items/second delivered by the reconstruction.
- **`ratio`** — delivered throughput ÷ the taught answer's throughput. `1.0`
  means "as good as what it was shown". This can exceed 1.0.
- **`beat`** — how many reconstructions *out-delivered* the answer they were
  taught. Unreachable before an ambiguous family existed; on `ASSEMBLER_BANK` a
  model shown a 1-line bank that builds 3 lines scores `ratio = 3.0`.

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
- **`ambiguous tasks: 0` everywhere** *(at the time of that run)*. Each
  conditioning had exactly one valid answer, so `functional == exact` is a
  property of the *data* — the two metrics moved together for 5,000 steps
  because getting it right and getting it working were the same event — and a
  30-line BFS beats the model at the task as posed.
- **`exact=1.000` came from n=64.** For an all-successes run the 95% lower bound
  is `0.05^(1/n)`: 64/64 proves only >95.4%, and per-lesson 16/16 only >82.9%.
  The fresh (held-out) training batches put the real entity error at **0.19%**
  and show `place < 1.0` on **16.8% of batches** — a tail the frozen set is too
  small to contain.

**Mitigations in place:** from-scratch validation (`Sample::blank_to_scaffold`)
masks everything but the source/sink, so the model must *design*, not inpaint;
`val_batch` default 64 → 512; `functional` is now item-aware; and
**`ASSEMBLER_BANK` breaks the ambiguity floor** — every one of its tasks admits
3 valid answers, delivering 1×/2×/3×:

```
ASSEMBLER_LINE   distinct factories:    135 | distinct tasks:    135 | ambiguous tasks:  0
ASSEMBLER_BANK   distinct factories:    135 | distinct tasks:     45 | ambiguous tasks: 45
```

Re-derive with `cargo run --release --example task_space`; see one task and its
three answers with `cargo run --release --example ambiguity_demo`.

**Giving the machines their real size made this worse, and that is worth stating
plainly.** A bank of three assembler lines is 7×9 once the assemblers are 3×3
instead of 1×1, and a 7×9 box has far fewer placements in an 11×11 grid than the
fictional narrow one did: the family fell from **189 tasks to 45** (~169× → ~711×
seen per task), and `ASSEMBLER_LINE` from 231 to 135. The ambiguity is untouched
(45 of 45, still 3 answers each), so what step 4 below bought is intact — but the
templated families are now *more* memorizable than the numbers quoted above them,
and the honest fix is not to shrink the machines back. **It is that size 11 is too
small a canvas for a real 3×3 machine**, which is the same conclusion bottleneck 0
reaches from the other direction. Grid size is the knob; see the open half of
step 4.

A caution learned the hard way here: `Sample::blank` observes every cell it does
not blank, so `removable` must list the region an answer *may* build, not the
cells a given answer *did* build. Listing only the built cells leaves an unbuilt
line observed-as-empty, which silently states the answer in the conditioning and
returns ambiguity to 0. Any new ambiguous family must be checked under `blank`,
not only under `blank_to_scaffold`.
**Next:** the remaining four families are still rigid. `move_one_item` is the
valuable one to fix (~42k tasks, honest scale) — its BFS picks one shortest path
where many exist, so the model is trained to imitate a tie-break. Randomizing it
should be measured, not assumed: same-conditioning collisions are rare at that
scale, so it may not move `ambiguous` much while risking mode-averaging.

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

### 2. Simulator fidelity — ✅ the metric can now rank two working factories
`item_reaches_sink` was a *binary* reachability check: it could not say which of
two working layouts was better, so Best-of-N had nothing to sort by and RL had
no gradient to climb. This was the blocker for almost everything downstream.

**Fixed.** `src/throughput.rs` scores a factory in items/second per sink, folded
by `((1/N)·Σ achievedᵢ^p)^(1/p)` at `p=0.5` so starving any one sink is punished
harder than slowing all of them. Flow propagates by Kahn's algorithm over a
graph whose edge `p → q` exists only if `p` pushes into `q` *and* `q` accepts
from `p`. Three deliberate departures from the reference, each a test in that
file:

- **The assembler is a real machine.** The reference never reads `crafting_time`
  or `crafting_speed` — it models a machine as a pass-through *ratio* capped at
  1.0, which reinterprets a per-craft count as a per-second rate. Right for 0.5 s
  recipes, 12–20× too generous for long ones, and it means a machine can never
  be the bottleneck — which is exactly what a machine usually *is*. We cap at
  `Recipe::crafts_per_second`.
- **Cycles degrade locally.** The reference scores the whole factory 0 if a cycle
  exists anywhere, even in a disconnected corner — a cliff, as a training signal.
  Here a cycle simply never gets a topological turn, so it starves what is
  downstream of it while sinks fed by other paths still score. Kahn's algorithm
  gives this for free; no cycle check needed.
- **No lanes — and the roadmap used to ask for them anyway.** The reference
  splits each belt tile into left/right lane nodes to model sideloading. That is
  vacuous *here*: an inserter has exactly one pickup tile, and belt merging is
  already handled by the per-tile cap. Porting lanes would have added nodes that
  can never differ. The real limitation is the **world model** (no lanes, no
  sideloading), not the throughput port — so "lane-aware throughput" was the
  wrong next step and is not one now. If we want lanes, they belong in
  `world.rs` first, and bottleneck 4 is where that lives.

Also fixed earlier: `item_reaches_sink` was **item-blind**, scoring "belt raw
plate straight into a gear sink" as functional — i.e. rewarding *skipping* the
assembler. It now carries the item through the BFS and applies recipes.

### 3. Receptive field / global routing
Addressed architecturally via the global-context vector, but for large grids a
single mean-pool may be too coarse.
**Next:** multi-scale U-Net (down/up sampling) or axial/attention blocks; measure
whether functional-rate scales with grid size.

### 4. Curriculum breadth & realism
Four hand-built lessons exercise every channel but are small and templated. Real
Factorio layouts are richer (multi-input recipes, buses, furnaces).

Machines are now the size they are in Factorio — an assembler covers 3×3, a
splitter 2×1 — stored at their top-left anchor with the rest of the footprint
`Empty` but claimed (`Grid::anchor_at`, `Grid::footprints_are_legal`). This was
not cosmetic: `blueprint.rs` had always exported a *real* 3×3
`assembling-machine-1` at the assembler's cell, so while the world model kept
1×1 machines every `ASSEMBLER_LINE` blueprint we emitted placed the machine on
top of its own inserters and Factorio refused the import
(`experiments/overlap_check.rs` reproduces the collisions). The model was being
taught a shape that cannot be built.

The footprint also unblocks the recipe simplification. Every `Recipe` here names
a *single* ingredient — our electronic circuit needs only an iron plate, where
vanilla needs 3 copper cable **and** 1 iron plate. That used to be forced by
geometry: a 1×1 machine has one tile in front and one behind, so there is
nowhere to put a second input. A 3×3 machine has twelve perimeter slots
(`Grid::perimeter`) and can be fed from as many sides as a recipe needs. What
still stands in the way is two things, neither of them the world: `Recipe`'s
single-`ingredient` field, and `sim.rs`'s reachability check, which walks a
single carried item and so cannot say "this machine runs only once *both* inputs
arrive". `throughput.rs` already can — it propagates a per-item flow vector and
sums every predecessor.

**Next:** multi-ingredient recipes (a real electronic circuit), a 2×2 furnace to
prove the footprint machinery is not 3×3-shaped, branching buses, curriculum
weighting by difficulty, and held-out lesson kinds to measure generalization.

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
   that matters. We took the *discipline* from the reference's `thput_eot`, not a
   number to beat: their ~0.11 is a **graded throughput** score and `functional`
   is **binary**, so the two are not comparable. `SCRATCH ratio` is the
   comparable one (and even then the world models differ).
2. **Graded throughput** ✅ *(this branch)* — `src/throughput.rs`, power mean at
   `p=0.5`. What makes one working factory rankable against another; everything
   below was blocked on it. Dropped the "lane-aware" qualifier: a world without
   lanes cannot tell two lane nodes apart (see bottleneck 2).
3. **Best-of-N sampling, verified by the simulator** ✅ *(this branch)* —
   `src/best_of_n.rs` and `sample --best-of N --temperature T`. Draw N layouts,
   simulate each, keep the best; needs **no retraining** because the sampler is
   already stochastic. `--blueprint-out` exports the winner, so the pipeline is
   generate → verify → best-of-N → export. `BestOfN::distinct` is the honest
   probe: if it stays at 1, the model holds one memorised answer and no larger
   `N` will help.
4. **A curriculum that admits many answers** ✅ *(this branch)* —
   `ASSEMBLER_BANK`: 3 sources and a shared sink are the task, and how many of
   the 3 assembler lines to build is the answer. All 45 tasks admit all 3
   answers, delivering 1×/2×/3×. This is what gives steps 2 and 3 something to
   do and what makes `beat_original` reachable at all.
   **Still open:** the other four families remain rigid, and the bank is a small,
   memorizable family — and giving the assemblers their real 3×3 footprint shrank
   it further, from 189 tasks to 45 (~711× each). **Raising the grid size is now
   the prerequisite**, not a nice-to-have: a 7×9 bank barely fits an 11×11 board,
   so there is nowhere left to put it. The next ambiguous family should be at
   `move_one_item` scale — multi-source/multi-sink, several recipes, tighter
   obstacle budgets — on a canvas big enough to hold real machines.
5. **Tune the imbalance knobs** — sweep `structure_weight`, add focal loss,
   compare mean-CE vs `--elbo`.
6. **Cheap architecture wins from the reference** — 1×1-conv tile head → softmax
   over the flat board (their PR #16: 2.6M → **520 params**, no throughput loss,
   +76.4% SPS); per-tile conditioned attribute heads `P(tile)·P(attrs|tile)`.
7. **Factorio parity** — RCON harness (1800 warmup / 3600 measure ticks at 32×)
   to prove the simulator is not lying. Not CI-able; needs a licensed install.
8. **RL/self-improvement (still last, and still not yet)** — the three
   preconditions it was waiting on are now met: throughput is graded (2), the
   sampler can be ranked (3), and at least one family admits many answers (4).
   That is *necessary* but not sufficient, and RL should still not be next:

   - **Best-of-N has now been spent, and it delivered.** On a checkpoint trained
     with `ASSEMBLER_BANK` in the mix, `--best-of 16 --temperature 1.0` beats
     greedy by **+48.7%** throughput for zero training cost and zero risk of
     collapse, and finds **4 layouts in 128 that are better than the generator's
     own answer** (greedy finds none). `BestOfN::distinct = 8.21` per task, so a
     policy gradient *would* have a distribution to sharpen — but the free method
     is already taking the gain, so RL now has to clear "better than 16 forward
     passes" rather than "better than nothing". See
     [`RL_ANALYSIS.md` §3.2](RL_ANALYSIS.md).
   - **One ambiguous family out of five is a thin base.** RL would optimise
     throughput on `ASSEMBLER_BANK` — 45 memorizable tasks — and could simply
     memorise "always build 3 lines" without learning anything about design.
     Widen the ambiguous curriculum first (step 4's open half).
   - **The simulator has not been parity-checked** (step 7). RL optimises the
     reward it is given, exactly and remorselessly. Handing it an unverified
     simulator means it will find that simulator's bugs rather than good
     factories — the standard failure mode, and much harder to notice than a
     crash.

   When it does happen: reward stays **terminal-only**. The reference *tried*
   potential-based shaping and rejected it (−2.8% thput at p=0.560, −18.3% SPS).
   The natural first form is not PPO but the cheapest thing that works —
   rejection sampling / expert iteration: run Best-of-N, keep the winners, fine-
   tune on them, repeat. It reuses the machinery in (2) and (3) exactly as-is,
   has no new hyperparameters, and cannot collapse the way a policy gradient can.

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

# Best-of-N: draw 16 candidates per task, keep whichever the simulator ranks
# highest, and export that one. Needs --temperature: greedy decoding draws the
# same factory every time and the extra passes would buy nothing.
cargo run --release --bin sample -- --ckpt checkpoints/denoiser \
  --best-of 16 --temperature 1.0 --blueprint-out generated-blueprint.txt

# Measure the curriculum itself (no model): how many tasks, how many answers each
cargo run --release --example task_space

# See one task and every valid answer to it, with the rate each delivers
cargo run --release --example ambiguity_demo
```
