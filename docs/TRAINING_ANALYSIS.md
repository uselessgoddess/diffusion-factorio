# Analysis of the 5,000-step GPU run

This answers the questions in [issue #5](https://github.com/uselessgoddess/diffusion-factorio/issues/5)
against the run that was posted there (wgpu, 11×11, seed 7, 5,000 steps). Every
number below comes from the `report.html` embedded JSON of that run, or from
`cargo run --release --example task_space` **as the curriculum stood at the time
of that run**.

That last qualifier is load-bearing now. Giving the assembler the 3×3 footprint
Factorio actually enforces (issue #9) cut `assembler_line` from 231 distinct
tasks to 135, and adding a fifth family changed how often a 5,000-step run sees
each one. The curriculum figures below are therefore a **historical record of
what this run trained on**, not a description of what `task_space` prints today;
where the two differ the text says so. `docs/ROADMAP.md` carries the current
numbers.

## TL;DR

The run is healthy and the model really did learn something — but the headline
`exact=1.000 functional=1.000` says much less than it looks like, and the
*evaluation could not have told us otherwise*.

1. **The last log line is not a bug.** `step 5000 | loss 0.3474 | place 0.91` is
   the single worst batch out of the last 1,000. At `lr 3.08e-11` the weights had
   not moved for hundreds of steps, so nothing regressed.
2. **`exact=1.000` is measured on 64 factories.** That is consistent with a true
   success rate as low as **83%** (95% Clopper-Pearson bound). The fresh training
   batches, which *are* held-out data, put the real entity error at **0.19%** —
   nonzero, and invisible to a 64-sample eval.
3. **The task is tiny.** On `assembler_line` the model is asked to fill in
   **2.0 cells out of 121**, both always `Inserter, East`, from one of only **231
   distinct templates**, each seen ~173 times. That is memorization scale.
4. **No task has more than one valid answer** (`ambiguous tasks: 0` in all four
   families). So `functional == exact` is a property of the *data*, and a 30-line
   BFS beats the model at the task exactly as posed.
5. **The model has never designed a factory**, because the scaffold is always
   given. This branch adds the eval that actually asks it to
   ([from-scratch validation](#the-fix-from-scratch-validation)).

The model's value only begins where BFS stops: when many layouts are valid and
you want the *best* one.

## "Why such a long training process?"

It wasn't long. From the run's own report:

```
total elapsed:            140.5 s   (2.3 minutes)
samples/sec overall:      1138.9
median pure train step:   27.76 ms
median validation step:   82.4 ms   (n=25)
total time in validation: 2.1 s     (1.5% of the run)
```

The whole thing took **2.3 minutes**, and validation — the part that looks
expensive — was 1.5% of it. So "long" can only mean *step count*, and the
gradual curve has a mundane cause: **the LR schedule, not the difficulty.**

Metrics saturate around step ~3,000, but cosine decay keeps running to 5,000.
The last ~2,000 steps ran at an LR small enough that the weights were frozen —
**~40% of the run was spent not learning.** The gentle approach to 1.0 is the
cosine tail, not the model slowly grasping something hard.

Two consequences:

- **Efficiency:** cutting to ~3,000 steps costs nothing measurable.
- **Quality:** the *reason* there was nothing left to learn is that the task is
  2 cells wide. The schedule is not the bottleneck; the curriculum is.

## Is the model actually learning, or memorizing?

`cargo run --release --example task_space` measures the curriculum with no model
involved. At size 11, 200,000 generator seeds per family, **as the four families
stood during this run**:

| lesson | distinct tasks | seen per 5k-step run | ambiguous |
|---|---:|---:|---:|
| `move_one_item` | 41,857 | ~1.0× | 0 |
| `move_one_item_chaos` | 200,000+ | ~0.2× | 0 |
| `assembler_line` | 231 | ~173× | 0 |
| `underground_cross` | 110 | ~364× | 0 |

(Today the same command prints **90** for `assembler_line`; the exact exposure
counts shift as the expanded curriculum shares the step budget, while the
assembler is 3×3 and vanilla recipes leave a single-source line only
two of the three recipes to roll. None of it rescues the argument below: 90
templates seen ~296× each is *more* memorizable than 231 seen ~173×.

Worse, even 90 flatters it. `task_space` now also counts factories with
translations collapsed, and `assembler_line` has **2** distinct shapes — the two
recipes it can roll. The other 45× is the same template at another offset, which
a fully-convolutional denoiser generalizes over for free. The real number is 2,
and it is 2 on a 19×19 board too. See `docs/ROADMAP.md` bottleneck 0.

An earlier `task_space` pass counted only **1** answer because it used the old
`Sample::blank` contract: the protected assembler and recipe remained visible,
so the model was never asked to predict them. That diagnostic exposed the
training bug. Production training and `task_space` now both use source/sink-only
conditioning; the recipe is an answer, and `assembler_line` correctly has **2**
task-conditioned answers. `ASSEMBLER_CHAOS` produces more than 150 in 200 seeds.)

**The curriculum splits cleanly in half.**

- `move_one_item` / `move_one_item_chaos`: each task is seen about **once**.
  Scoring 1.000 here is **real generalization** — the model learned to imitate a
  deterministic BFS router, tie-breaking (N, E, S, W) included.
- `assembler_line` / `underground_cross`: a couple hundred templates seen
  hundreds of times each. 1.000 here is **memorization**, and it is what makes
  the aggregate number look perfect.

`assembler_chaos` has since moved the assembler half across the line: its
source, sink, recipe, and obstacles define a canonical machine pose and route,
and more than 150 task-conditioned answers appear in only 200 seeds.
`underground_cross` is still on the memorizing side, and this table is the reason
to expect the aggregate to *drop* when the rest follow: the run that produced the
numbers in this document was scoring a curriculum half of which it had memorized.

Both halves are worth knowing. The first is a genuine (if modest) result. The
second is why the aggregate should not be trusted.

### And the masked cells are few

Cells the model must fill, out of 121:

```
assembler_line        2.0 masked cells/sample  (1.7% of the grid)
move_one_item         6.3 masked cells/sample  (5.2%)
move_one_item_chaos   6.7 masked cells/sample  (5.5%)
underground_cross     4.0 masked cells/sample  (3.3%)
```

The source, sink, assembler, recipe tag and obstacles are **always given**. The
model is doing inpainting, not design.

## The step-5000 "anomaly"

Over the last 1,000 steps, with the model effectively frozen:

```
loss  p50 0.0073 | p90 0.0418 | p99 0.1411 | p100 0.3474
steps with loss > 0.10:  26/1000 (2.6%)
steps with loss > 0.30:   1/1000 (0.1%)
```

`0.3474` **is** that p100 — the log's final line landed on a 1-in-1000 hard
batch by chance. `place 0.91` on it is the true residual error surfacing.

That residual is the interesting part. Training batches are freshly generated
every step (`seed_ctr` increments, never repeats), so they are held-out data:

```
~1,967,454 masked cells evaluated, ~3,705 wrong  -> entity error rate 0.1883%
mean placement recall 0.9952                     -> 0.48% of structure cells wrong
steps where entity_acc < 1.0 on a fresh batch:   725/1000 (72.5%)
steps where place      < 1.0 on a fresh batch:   168/1000 (16.8%)
```

**The model fails on 16.8% of fresh batches.** The frozen 64-factory validation
set reports 1.000 because it is too small to contain the tail.

### Why n=64 cannot see it

For an all-successes run, the 95% lower bound on the true rate is `alpha^(1/n)`:

```
 16/ 16 perfect -> true rate could be as low as 0.829
 64/ 64 perfect -> ...................... 0.954
1000/1000 perfect -> .................... 0.997
```

Per-lesson (n=16) the run's `exact=1.000` is compatible with a true rate of
**83%**. This branch raises the `val_batch` default 64 → **512**, which lifts the
aggregate bound to 0.994 and the per-lesson (n=128) bound to 0.977.

That is not free, and the honest accounting is: 8× the factories, now scored in
2 modes, is 16× the validation work. On the posted run that is 2.1 s → ~34 s,
turning a 140 s run into ~170 s — validation goes from 1.5% to ~20% of wall
clock. Worth it: the alternative is a headline number that cannot distinguish a
perfect model from an 83% one. CI overrides it (`--val-batch 32`) to stay fast.

## The metric bug this uncovered

`functional` was **reachability-only and item-blind**: it asked "does flow reach
the sink?", not "does the *right item* arrive?". So belting raw iron plate
straight into a gear sink scored as functional — i.e. the metric rewarded
skipping the assembler. Fixed in `27867b3`; `item_reaches_sink` now carries the
item through the BFS and applies recipes. The reference guards the same hole
(`throughput.rs:205-226`, "sinks only score their configured item").

This mattered little while the assembler was always given to the model. It
matters entirely once the model has to build one.

## The fix: from-scratch validation

`Sample::blank_to_scaffold()` masks everything except the source and sink.
Obstacles stay — they are terrain, not something the model places. The model is
told only *"plates enter here, gears must arrive there"* and must decide what to
build and where. Now ~119 of 121 cells are masked instead of 2.

This mirrors the reference's honest `thput_eot` metric, which blanks the whole
grid and rebuilds from empty. It is reported alongside the old metric
(`|| SCRATCH ...` in the progress line, its own chart and per-lesson table in
the HTML report), so the inpainting number stays comparable across runs.

**Read `functional`, not `exact`, in this mode.** Many layouts deliver the item;
`exact` only rewards rediscovering the generator's own BFS answer, so it
understates. This is also the first metric in the project where `functional` and
`exact` genuinely come apart — under the old eval they were forced to agree.

## Where the dataset comes from in the original

**It is 100% procedural. Nothing is scraped, and no human blueprints are used.**
This project and the reference agree here, which is a strong signal the approach
is right.

In `beyarkay/factorion` (HEAD `fdb723a`):

- A Rust generator (`factorion_rs/src/factory_gen.rs`) builds factories by
  randomized rejection sampling — `build_factory(size, kind, seed, random_item,
  max_entities)`, 13 lesson variants, 11 active.
- **The throughput engine is also the data validator.** A candidate is only
  emitted if the simulator says it works. Same principle as our
  `item_reaches_sink` check at generation time.
- Data is **streamed online**, never materialized: `StreamingDemoDataset`
  (`sft.py:323-350`), `num_samples = 45_000_000` with `epochs = 1`.
- Train/val are disjoint **by seed arithmetic** (`sft.py:932-946`) — same trick
  as ours.
- They **balance lessons on emitted (state, action) pairs**, not on factories,
  because big factories emit ~10× more pairs (`sft.py:263-266`).

Two things they do that we should copy:

1. **Scale.** 45M samples, single epoch. Our 5k × 32 = 160k samples over a
   curriculum with ~240 distinct tasks in half its families is a different
   regime entirely. Our data generator runs at 13,811–22,787 gen/s (~1.5–2.3 ms
   per batch of 32) against a 27.76 ms train step — **data generation is only
   ~7% of step time and is not the bottleneck.** Batch 32 underutilizes the GPU;
   there is a large free win in raising it.
2. **Lesson balancing on the unit you actually train on**, once our lessons
   differ in size.

They also **explicitly reverted a held-out-recipe split** (#272), on the grounds
that "memorising recipes is desired behaviour". Worth knowing before we treat
recipe memorization as a failure.

## What about RL for maximum throughput?

**Yes — but it is step 3, not step 1, and the prerequisite is a graded reward.**

Right now RL is *impossible* here, and not for a training-infrastructure reason.
Every task in the curriculum has exactly one correct answer (`ambiguous: 0`), so
the reward is binary and already saturated at 1.0. **There is nothing for a
policy gradient to climb.** RL needs a metric that says one working factory is
*better* than another working factory — that is throughput, and we do not have
it yet.

What the reference learned about RL, which we get for free:

- **Reward is terminal-only; no shaping** (`ppo.py:883-888`). They *tried*
  potential-based shaping and **rejected it**: −2.8% throughput (p=0.560, i.e.
  not even significant) for −18.3% SPS (PR #18).
- **SFT and PPO share one `nn.Module`.** PPO warm-starts from the SFT weights;
  `--critic-warmup N` freezes the actor while the value head catches up
  (`ppo.py:1999-2008`), which is the standard fix for the SFT→PPO handoff
  destroying a good policy in the first few updates.
- Their tuned HPs, if we get there: `learning_rate=3.369e-05, gamma=0.9566,
  gae_lambda=0.9187, clip_coef=0.1987, ent_coef 0.008034 → 0.0007372,
  vf_coef=0.794, critic_warmup=9, target_kl=0.02`.

**But note where they actually are.** Per their `CLAUDE.md` (the only doc that
matches their code — `README.md` and `docs/EXPERIMENTS.md` both contradict it and
are ~8 months stale), the canonical SFT base scores `val/thput_eot ≈ 0.11`:
`move_one_item ≈ 0.38`, and **the assembler lessons ≈ 0**. With RL, PPO, 45M
samples and a full throughput engine, **they cannot yet build a working assembler
factory from scratch.** That is the bar. It is lower than their README suggests,
and it is the honest thing to measure ourselves against.

So: build the graded metric first, see what plain sampling gets, and only reach
for RL when there is a gradient to climb.

## What to borrow from the reference

Ranked by value per unit of work.

1. **Throughput as a graded score.** `score = ((1/N)·Σ achievedᵢ^p)^(1/p)` with
   `p = 0.5` (`throughput.rs:7-17,39-45`) — the power mean punishes starving any
   one sink, which a plain average does not. Lane-aware flow graph
   (`graph.rs:88-91`); rates (`types.rs:510-521`): belt 15.0, inserter 0.86,
   long inserter 1.2, assembler-1 0.5, underground 15.0, splitter 15.0,
   source/sink ∞, per-lane cap `/2` → 7.5. **This is the unlock**: it makes the
   objective graded, which makes best-of-N meaningful and RL possible at all.
2. **1×1-conv tile head → softmax over the flat board** (`ppo.py:1243`). Their
   PR #16 cut the head from 2.6M to **520 parameters (~5000×)** with **no**
   throughput loss and **+76.4% SPS** (p=4.7e-09). Nearly free efficiency.
3. **Per-tile conditioned attribute heads** — `P(tile) · P(attrs | tile)` rather
   than independent per-channel heads. Directly relevant to our joint-consistency
   problem.
4. **Their honest metric discipline** — `thput_eot` blanks the whole grid. We
   have now adopted this.
5. **RCON parity harness** (`factorio-mod/README.md`) — sources/sinks as real
   scripted belts, 1800 warmup / 3600 measure ticks at 32× speed, e.g.
   `engine 15.000/s, factorio 14.870/s (err 0.9%) ok`, verified on Factorio
   2.0.76. This is how you prove the simulator is not lying. Not CI-able (needs a
   licensed install), but worth having.

**Do not borrow:**

- **Their assembler model is wrong.** `entities.rs:426-451` never reads
  `crafting_time` or `crafting_speed` and caps at `min_ratio ≤ 1.0`. A real
  assembler-1 is 0.5 crafts/s with per-recipe times; theirs is a pass-through
  ratio. If we implement throughput, implement it correctly.
- **`thput_normed` needs the scripted reference solution** to normalize, so it
  only exists for lessons they can already solve by construction — a live
  `FIXME(#161)` at `ppo.py:654-665`. Our power-mean score should be absolute.
- **Argmax-only inference** — it can loop on a tile. Our confidence-ordered
  reveal is better, and best-of-N is better still.

Their global mean+max pooled context (`ppo.py:1245-1258`, #290) is a good idea
that **this repo already has** (`src/model.rs`).

## "I want to generate real useful schemes that weren't baked in"

This is the right thing to ask for, and it is the thing the old eval could not
answer. The gap, concretely:

- The model has only ever seen 11×11 grids with one source and one sink.
- It has only ever been asked to fill 2–7 cells of a given scaffold.
- Every question it was asked had exactly one right answer, computable by BFS.

So the ranked path to "useful schemes it wasn't taught":

1. **From-scratch eval** (this branch) — establishes the real baseline. Expect
   `exact` to collapse and `functional` to become the number that matters. We
   should expect this to hurt; the reference gets ~0.11 on the equivalent.
2. **Graded throughput** (borrow #1) — the objective stops being binary.
3. **Best-of-N sampling, verified by the simulator.** The single highest-leverage
   step for *usable output*: sample N layouts, simulate each, keep the best. Our
   diffusion sampler is already stochastic (temperature) and already conditional,
   so this needs no retraining. It converts "the model is right 99.8% of the
   time" into "the exported blueprint works", and it stacks with everything else.
   Pipeline: **generate → verify → best-of-N → export**.
4. **Curriculum that admits many answers** — multi-source/multi-sink, several
   recipes, tighter obstacle budgets. Until a task has more than one valid
   solution, the model cannot demonstrate design and BFS remains the better tool.
5. **Only then RL** for squeezing throughput.

## Honest summary

The concept is not yet proven, and the run posted in the issue does not prove it
— but nothing in it is broken either.

What is genuinely established: the model learned to imitate a deterministic BFS
router on two families whose task space is far too large to memorize. That is a
real result, and it means the architecture and the training loop work.

What is not established: that it can *design* anything. It has never been asked
to. On the two families where it scores 1.000 by memorizing ~240 templates, and
on the 2-cell inpainting task, a BFS is strictly better than a neural network.
The model earns its keep only when there are many valid answers and we want the
best one — which requires the graded throughput metric, and which is exactly
where the reference is still stuck at ~0 on assembler lessons.

The next commit that matters is not a bigger model or a longer run. It is a
metric that can tell two working factories apart.
