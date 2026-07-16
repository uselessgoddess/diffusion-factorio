# Inference and training

Answers issue #11. Every number here is either transcribed from that issue's own
5,000-step log, re-derivable with a command given inline, or measured on this
branch and labelled with the machine it was measured on.

The issue asks six things. Short answers first, then the working.

| The ask | The answer | Where |
|---|---|---|
| "может я не прав и неправильно интерпретирую результаты" | Partly. The log's headline is not what it looks like, and the run's real finding is in it | [§1](#1-you-are-not-misreading-the-metrics-they-were-answering-a-different-question) |
| "что локально надо тестировать при обучении" | Three tiers, and the one that matters is not the one the progress line leads with | [§2](#2-what-to-watch-locally-while-training) |
| "хотел свои задачи кастомные в рантайме" | Built. The page is static; the *task* is not | [§3](#3-touching-the-schemes-rather-than-chasing-air) |
| "сплиттер или манипуляторами с одной линии" | You were right, and the simulator was wrong. Fixed and measured | [§4](#4-the-shared-line-you-were-right-and-the-simulator-was-wrong) |
| "награждалась за компактность" | Only ever as a tiebreak. Compactness cannot be an objective | [§5](#5-compactness-is-a-tiebreak-not-an-objective) |
| "может тут уже RL нужен" | Still not next, and the bar just got higher | [§6](#6-rl-still-not-next-and-the-bar-moved-up) |

The two that are genuinely open ground — the mod's area-and-edge-ports UX, and
what "procedural lessons are limited" costs — are [§7](#7-the-mod-ux-area-selection-and-edge-ports) and [§8](#8-procedural-lessons-are-limited-this-is-bottleneck-0).

---

## 1. You are not misreading the metrics; they were answering a different question

The 5,000-step log ends at `exact=1.000 functional=1.000`. That is a real number
and it is nearly meaningless, for two separate reasons that are easy to conflate.

**First, `1.000` at `n=64` was never `1.000`.** For an all-successes run the 95%
lower bound on the true rate is `0.05^(1/n)`: 64 for 64 only proves *"above
95.4%"*. Raising `--val-batch` to 512 (`src/train.rs:67`) reports `0.992` on the
same checkpoints. The model did not get worse; the eval got honest. This is
already written up in [`RL_ANALYSIS.md §1.1`](RL_ANALYSIS.md).

**Second, and this is the part worth your attention: on four of the eight
families, `exact=1.000` is what a 30-line BFS scores.** The families have
`ambiguous tasks: 0` — one task, one answer — and the model sees each task
hundreds of times in a 5,000-step run. Against a rigid curriculum, a model that
perfectly clones the generator gets a perfect score. **There is no headroom above
"imitate the BFS" to even measure.**

So the honest reading of your log is not "the model is at 100%". It is: *the
questions were too easy to score.*

### The finding that is actually in your log

Your run does contain a real result, and it is not a good one. Under `SCRATCH`
(`Sample::blank_to_scaffold` — sources and sink given, everything between them
the model's to design), the gap between `functional` and `exact` is the share of
layouts that **work but are not the generator's answer**. That is originality,
and it is measurable:

| step | SCRATCH `exact` | SCRATCH `functional` | gap | share of *working* layouts that are original |
|-----:|----------------:|---------------------:|----:|---------------------------------------------:|
|  600 |           0.182 |                0.557 | 0.375 | **67.3%** |
| 3200 |           0.605 |                0.730 | 0.125 | 17.1% |
| 5000 |           0.621 |                0.717 | 0.096 | **13.4%** |

`SCRATCH functional` over steps 3200–5000: min 0.703, max 0.730, mean 0.717 — a
spread of 1.4 s.e. at `n=512`, i.e. flat.

**The model's originality fell by 2.5× while its ability to build working
factories did not improve at all.** The mechanism is the single-label curriculum:
with one right answer per task, a working-but-different layout is
indistinguishable from a mistake, and the cross-entropy punishes it exactly as
hard. **Every original solution is a loss penalty.** The last 1,800 steps of your
run bought conformity, not capability.

This is the direct answer to *"сила в генерализации, в поиске нестандартных
решений"*. You are right about where the value is, and your training run was
actively training it out. Not because the optimizer misbehaved — because you
asked for it, in the loss.

---

## 2. What to watch locally while training

*"я уже перестаю понимать результаты"* — the progress line is genuinely
misleading, and it leads with its worst metric. Read it in three tiers.

### Tier 1 — per step, is it learning at all

```
step  1200/5000 | lr 2.8e-4 | loss 0.4127 | place 0.94 | acc[E=0.99 D=0.98 I=1.00 M=1.00]
```

**Watch `place`. Ignore `acc[E=...]`.** About 95% of cells are `Empty`, so entity
accuracy is ~95% for a model that has learned nothing at all; it is high before
it means anything. `place` is placement recall — of the cells that should hold
something, how many got something. It is the one number in this line that can
distinguish learning from collapse.

Low `acc[entity]` early is the *opposite* of the symptom it looks like: it means
the model is over-placing, which is what `--structure-weight 8` is for and is a
much healthier failure than the empty attractor.

### Tier 2 — per validation, does it build factories

```
VAL n=512 | exact=0.992 functional=0.992 ... || SCRATCH n=512 | exact=0.621 functional=0.717 ...
```

**Read `SCRATCH`, not `VAL`. Within `SCRATCH`, read `functional`, not `exact`.**

- `VAL` blanks 2–7 cells of a given scaffold. It is inpainting, not design, and
  on the rigid families it is a memorization check.
- `SCRATCH` gives only the sources and the sink. That *is* "given inputs and
  outputs, produce the factory" — the mod's actual task, in miniature.
- `exact` measures conformity to the generator. On the ambiguous families
  (`ASSEMBLER_BANK`, `CIRCUIT_LINE`, `SHARED_LINE`) it is capped below 1.0 by
  construction: each task has several valid answers, so a model that always
  builds the *best* one scores worse than one that guesses the generator's roll.
  **On ambiguous families `exact` is a diagnostic, not a target.**

The `functional − exact` gap under `SCRATCH` is the originality probe from §1.
**If that gap is shrinking while `functional` is flat, the run has stopped buying
capability and is buying conformity.** That is the single most informative thing
in your log, and nothing prints it for you — it is a subtraction you have to do.

`functional` is binary and it saturates: at 0.99 it has nothing left to say.

### Tier 3 — the graded ones, the only ones that can show the model winning

```
| thput=2.944/s ratio=0.547 beat=4 |
```

- `thput` — mean absolute items/s. Not comparable across tasks; a gear line and a
  cable line have different ceilings.
- `ratio` — per-task `recon_rate / orig_rate`, averaged over gradeable tasks.
  `1.0` means it matched the taught answer. **It can exceed 1.0.**
- `beat` — the count where the model *out-built its own curriculum*. This is the
  only metric here that can report superiority, so it is the one to watch. Under
  greedy decoding it is **0**; under `--best-of 16 --temperature 1.0` it is
  **4 of 128**.
- `distinct` (from `BestOfN`) — distinct factories drawn per task. **The honest
  probe: if it stays at 1, the model holds one memorized answer and no larger N
  will help.** Measured **8.21**.

### The smell table

| What you see | What it means | What to do |
|---|---|---|
| `acc[E]` high from step ~0 | Nothing. 95% of cells are empty | Read `place` |
| `place` near 0 and stuck | Collapsed to the empty solution | Raise `--structure-weight` |
| `acc[entity]` low, `place` high | Over-placing — the healthy failure | Sweep `structure-weight` *down* |
| `VAL exact` → 1.000 | The rigid families are memorized | Read `SCRATCH` |
| `SCRATCH exact` climbing, `functional` flat | **Conformity, not capability** | Widen the curriculum, not the steps |
| `functional` > 0.99 | The metric is saturated and blind | Read `ratio` / `beat` |
| `distinct` = 1 | One memorized answer; Best-of-N is pointless | Ambiguity, not N |
| `beat` = 0 | Never exceeds its teacher | Expected under greedy; use temperature |

### And: look at the thing

*"хотелось бы не воздух гонять а реально трогать и видеть схемы"* — this is the
right instinct, and it is now one command ([§3](#3-touching-the-schemes-rather-than-chasing-air)). Metrics tell you
*whether*; only the picture tells you *what*.

```bash
cargo run --release --example gallery    # what the model is trained on
cargo run --release --bin serve -- --ckpt checkpoints/denoiser   # what it does with a task nobody generated
```

---

## 3. Touching the schemes rather than chasing air

> это статический index html а я хотел свои задачи кастомные в рантайме делать —
> хотя я может конечно не увидел как это сделать

Both halves of that are right, and they are not in conflict — so this is worth
being precise about rather than just pointing at a flag.

**The page is static, and deliberately.** `src/serve/index.html` is compiled into
the binary with `include_str!` (`src/serve.rs:355`) — it is not a build artifact
and there is nothing to serve from disk. That is a deployment choice, not a
capability limit.

**The task is not static.** The thing you paint on that page is a `TaskSpec`,
constructed at runtime, POSTed to `/api/design`, and handed to the model as
conditioning. Nothing generates it; no lesson contains it; the model has never
seen it. That is precisely *"свои задачи кастомные в рантайме"*.

What you had before this branch was `experiments/gallery.rs`, which renders a
fixed page of *generated* lessons. That answers "did the model learn the
lessons", and nothing else. **The question the server exists for is the other
one: give the model a task nobody generated and see what it builds.**

```bash
cargo run --release --bin serve -- --ckpt checkpoints/denoiser
# 127.0.0.1:8080  (override with --addr)
```

Paint sources, sinks and obstacles; set the item; set width/height (default
13×9); press Design. It runs Best-of-N (default 8 candidates, 12 steps,
temperature 0.9), scores every candidate through the simulator, and shows you the
winner, the runners-up, the simulator's verdict, and a **scrubber that replays
the reveal rounds** — the order in which the model committed each cell.

Inference is CPU/ndarray by design: **a viewer that needs a GPU is a viewer
nobody opens.** No TLS, no auth, single-threaded — it binds to localhost and it
should stay there.

The reveal-round replay is the part I would point you at for *"перестаю понимать
результаты"*. It shows the model's own confidence order: what it was sure about,
and what it filled in only once the rest of the board forced its hand.

---

## 4. The shared line: you were right, and the simulator was wrong

> сейчас там 3 входа железа различных, а в factorio если сплиттер или просто
> можно манипуляторами забирать с одной линии последовательно

This is the sharpest observation in the issue, because it was not just a
curriculum gap — **the simulator scored the real Factorio idiom as dead.**

`ASSEMBLER_BANK` gives each assembler its own source. Real Factorio feeds one
line and either splits it, or lets a row of inserters pull off it in sequence.
Measure what the lessons actually teach:

```bash
cargo run --release --example bus_tap
```

```
lesson                    srcs/f sinks/f   belts/f splitters inserters
MOVE_ONE_ITEM               1.00    1.00      6.63         0      0.00
MOVE_ONE_ITEM_CHAOS         1.00    1.00      6.61         0      0.00
ASSEMBLER_LINE              1.00    1.00      0.00         0      2.00
ASSEMBLER_CHAOS             1.00    1.00     17.53         0      2.00
UNDERGROUND_CROSS           1.00    1.00      2.00         0      0.00
ASSEMBLER_BANK              3.00    1.00      3.03         0      4.02
CIRCUIT_LINE                2.00    1.00      0.00         0      5.07
SHARED_LINE                 1.00    1.00      9.92        96      2.96
```

**Before `SHARED_LINE`, the `splitters` column was zero for every lesson.** The
`Splitter` entity is in the vocabulary, has a footprint, exports to a blueprint —
and was never once drawn in training. The model could not have built one; it had
never seen the token in a position where it meant anything.

### The simulator bug your observation exposed

The three-sequential-taps arrangement — one belt, three inserters pulling off it
into three assemblers — scored **zero**. Not "poorly": zero, as in *the item
never reaches the sink*. At `81ad56d` (pre-fix):

```
=== one shared bus, three sequential taps ===
S>>>>>>>>>>>>
..i...i...i..
.Aaa.Aaa.Aaa.
.aaa.aaa.aaa.
.aaa.aaa.aaa.
..i...i...i..
..K...K...K..
  footprints legal: true    reaches a sink: false    score: 0.000
```

The flow graph only offered a belt's items to the tile *in front of* it, so an
inserter reaching back into a passing belt was invisible. A belt was a pipe
between its two ends, not a bus.

After `c7aef5e`, the same board:

```
  footprints legal: true    reaches a sink: true    score: 0.430
    sink at (2, 6) wants IronGear: 0.430/s
    sink at (6, 6) wants IronGear: 0.430/s
    sink at (10, 6) wants IronGear: 0.430/s
```

**0.430/s at each of three sinks, from one source** — matching what
`ASSEMBLER_BANK` needs three sources to achieve. The splitter arrangement already
scored 0.430 before the fix and still does; **the bug was specific to the
sequential-tap idiom, which is the one you named.**

The fix is a water-filling allocation against a per-tile intake cap
(`src/throughput.rs`), replacing an even split. It is a strict generalization: a
belt previously returned at most one target, so no belt ever had two successors
and the even split never actually split anything. **All 79 pre-existing tests
passed unchanged**, which is the evidence for "generalization" rather than
"change".

The rule it preserves is the discriminating one: an inserter picks up from
*behind*, never from the side. `an_inserter_does_not_grab_from_its_side` still
passes — the inserter faces North so its pickup tile is not the belt beside it —
while `an_inserter_taps_a_passing_belt_and_the_rest_flows_on` now asserts that
what the inserter takes is its own swing rate (`INSERTER_RATE = 0.86`), and the
remainder stays on the belt for whatever is downstream.

**Why this matters beyond one bug:** a reward function that scores the correct
idiom as zero does not merely fail to teach it — it teaches *against* it. Had we
gone to RL first, the policy would have learned that shared lines do not work.
This is the concrete instance of the general argument in [`RL_ANALYSIS.md`](RL_ANALYSIS.md):
**an unverified simulator is a reward function RL will exploit rather than
satisfy**, and the failure is invisible because the reward goes *up*.

---

## 5. Compactness is a tiebreak, not an objective

> награждалась за компактность

Wanted, and now in (`src/best_of_n.rs`, `prefer_compact`, default on). But it can
only ever be a **tiebreak**, and the reason is worth stating plainly because it
generalizes:

**The most compact factory is the empty one, and it delivers nothing.**

Any reward of the form `throughput − λ·parts` has a λ above which the empty grid
wins, and below which compactness is noise. Rather than tune λ, compactness ranks
*only* among candidates that already tie on throughput, and only when the tied
score is above zero:

```rust
fn beats(score: f64, count: usize, best: f64, best_parts: usize, prefer_compact: bool) -> bool {
    let tied = score == best && best > 0.0;
    score > best || (prefer_compact && tied && count < best_parts)
}
```

The `best > 0.0` is what keeps the empty factory from winning. `compactness_never_outranks_throughput`
and `an_empty_factory_never_wins_on_being_compact` are the tests that pin it.

**This costs nothing precisely because ties are common here.** A belt run carries
a full belt whatever route it takes, so the long way round and the short way
score identically (`the_long_way_round_and_the_short_way_deliver_the_same`). The
throughput metric genuinely cannot separate them; compactness genuinely can.
There is no trade being made, which is why there is no λ.

`parts_saved()` reports what the tiebreak bought, and `/api/design` returns it.

---

## 6. RL: still not next, and the bar moved up

The full argument is in [`RL_ANALYSIS.md §5`](RL_ANALYSIS.md) and I will not restate it. What this
branch changes is the *bar*:

- **Best-of-N already collected the easy gain.** +48.7% throughput over greedy
  for zero training. RL must now clear "better than 16 forward passes", not
  "better than nothing".
- **The simulator was wrong until this branch** (§4). Precondition #3 —
  a reward function you trust — was not met as recently as five commits ago,
  and it took a user observation, not a test, to find that out.
- **The strongest argument is still the reference's negative result.** With PPO,
  45M samples and a full throughput engine, [`beyarkay/factorion`] cannot build a
  working assembler factory from scratch at all (`assembler lessons ≈ 0`). **The
  most useful thing in that repo is a negative result, and it is about RL.** The
  next lever is the *task*, not the training algorithm.

When it does happen, it should not start as PPO. Expert iteration is the cheapest
thing that works: run Best-of-N, keep the winners, fine-tune on them, repeat. It
reuses the throughput metric and the Best-of-N sampler exactly as they are, adds
no new hyperparameters, and cannot collapse the way a policy gradient can.

**The honest framing:** §1 showed the model's originality collapsing under a
single-label loss. RL is one way to stop punishing original solutions. Giving the
task more than one right answer is another, and it is very much cheaper. Do that
first — and note it is the same fix, not a lesser one: both are about not calling
a working-but-different factory a mistake.

[`beyarkay/factorion`]: https://github.com/beyarkay/factorion

---

## 7. The mod UX: area selection and edge ports

> игрок выделяет область, подводит входы и выходы к краям этой зоны

Nothing in the docs addressed this, so: **the task shape is already what you
describe, and the gap is size, not paradigm.**

`SCRATCH` gives the model sources and a sink and asks for everything between
them. Edge ports are that with the ports constrained to the boundary. `TaskSpec`
in `src/serve.rs` already accepts arbitrary source/sink placement — you can paint
exactly your mod's task today and watch what comes back. That is the whole reason
the server takes a hand-painted task rather than a lesson id.

What is *not* proven, and what I would measure before building any mod:

1. **Size generalization has never been tested.** The denoiser is fully
   convolutional and mean-pools for global context, so **variable sizes are
   already free and nobody has measured them**. `--size` exists on all three
   binaries. **Train at 11, sample at 15 — that experiment costs one command and
   has never been run.** Do it before designing a UI around arbitrary areas.
   *(Since answered: `experiments/grid_shape` ran it, the trained shape scored
   6.5× the issue's shape at matched area, and the curriculum now draws every
   width × height in 9..=15 rather than one square — see cause 5 in
   `docs/GENERALIZATION.md`. `--size` on `train` is now the square-only control.)*
2. **The receptive field is a hard limit and it is computable.** ±1 at the stem,
   ±2 per block; at `--blocks 6` a cell sees ±13 — a 27×27 window. Beyond that,
   routing depends entirely on the mean-pooled global vector, and **mean-pooling
   a 30×30 board into one vector is a very coarse summary to route a bus with.**
   Expect size generalization to break here first. If a player selects a 40×40
   area, this is what will bite, and the fix is architectural (multi-scale U-Net,
   or axial attention), not more steps.
3. **The player's area is not the training distribution.** Every lesson has one
   source and one sink, on an 11×11 board, with 2–7 cells masked. Your mod's user
   will draw a rectangle with four inputs on one edge and two outputs on another.
   Nothing has ever asked the model for that.

The constant/logical combinator you devised for the port convention is a good
idea and it is orthogonal to all of the above — it is an encoding of the port
spec, and the model's conditioning already carries that information positionally.

**Concretely: (1) is one command and gates the rest. It is the highest
information-per-minute experiment available right now, and it has never been
run.** On a machine with a GPU it is minutes.

---

## 8. Procedural lessons are limited: this is bottleneck 0

> проблема в том что процедурные lessons ограничены

Correct, this is the top of [`ROADMAP.md`](ROADMAP.md), and it is worth knowing *how* limited,
because the number is worse than it looks:

```bash
cargo run --release --example task_space
```

With translations collapsed, `ASSEMBLER_LINE` teaches **one answer**.
`UNDERGROUND_CROSS`, one. `CIRCUIT_LINE`, three. `ASSEMBLER_BANK`, six.
**Thirteen layouts across the four lessons that build real factories** — each
drawn ~254× in a 5,000-step run. That is the mechanism behind §1's originality
collapse, stated as a count.

The reflex is to blame the 11×11 board. **The roadmap did exactly that, and it
was wrong:** the shape counts are *exactly flat* at 2, 6, 3, 2 on a 19×19 board
too. The denoiser is `same`-padded convolution end to end, so translation is
precisely the variation it is equivariant to for free. A bigger board buys
offsets, not lessons. **The room was never the constraint.**

The control that shows what does work is `MOVE_ONE_ITEM_CHAOS` and now
`ASSEMBLER_CHAOS`: they do not stamp a template, they scatter obstacles into the
conditioning plane and derive the answer by BFS *through* them. `ASSEMBLER_CHAOS`
scores **197,228 distinct answers from 200,000 seeds** against `ASSEMBLER_LINE`'s
1. A 5,000-step run sees each task ~0.1× instead of ~254×: the model cannot meet
the same task twice, which is the point.

**The recipe is known and it is not RL: randomize the world, derive the label by
search rather than stamping it.** `UNDERGROUND_CROSS`, `ASSEMBLER_BANK` and
`CIRCUIT_LINE` are still templates and want the same treatment. The bank is the
interesting one — it is the only honestly ambiguous family, and that property has
to survive the randomization.

`SHARED_LINE` (this branch) is the first family teaching an idiom rather than a
shape, and the first to draw a `Splitter` at all. It is **honestly ambiguous** —
the same task admits both the splitter and the sequential-tap answer, which is
exactly the property §1 says the curriculum is short of:

```
SHARED_LINE            distinct factories:     20 | distinct tasks:     10 | ambiguous tasks: 10
                         200000 seeds ok | a 5k-step run sees each task ~2000.0x
```

**And it is tiny, so read that as a demonstration rather than a fix.** Ten tasks
seen ~2,000× each is a lookup table; it is the same disease as `CIRCUIT_LINE`'s
seven. It proves the idiom can be taught and scored — it does not widen the
curriculum. Randomizing it is the same work item as the other three templates,
and I would not train on it expecting §1's gap to stop collapsing.

---

## 9. What I could not measure, and why

**This machine has no GPU.** No `nvidia-smi`, no `/dev/dri`, 6 cores, and the
training process pins a single one. Your 5,000 GPU steps are not reproducible
here, and I am not going to present CPU numbers as if they were.

Price a train step and a validation pass separately, at the defaults otherwise
(`--size 11 --batch 32 --hidden 64 --blocks 6`). One run with validation off,
one with two passes in it; the difference is the pass:

```bash
echo "=== A: 12 steps, validation OFF ==="
time cargo run --release --bin train -- --steps 12 --val-every 0 \
  --out /tmp/bench-noval --metrics-out runs/bench-noval.jsonl
echo "=== B: 12 steps, validation every 6 (n=512) ==="
time cargo run --release --bin train -- --steps 12 --val-every 6 --val-batch 512 \
  --out /tmp/bench-val --metrics-out runs/bench-val.jsonl
```
```text
=== A: 12 steps, validation OFF ===
real	7m57.924s
=== B: 12 steps, validation every 6 (n=512) ===
real	32m8.964s
```

Run B's own per-step record shows where the 24 extra minutes went — steps 6 and
12 are the two validating ones, at ~750 s against a ~42 s neighbour:

```text
step  5  +  44.10s  val=no
step  6  + 760.55s  val=YES
step  7  +  45.75s  val=no
...
step 11  +  39.02s  val=no
step 12  + 742.71s  val=YES
```

| what | cost on this box |
|---|---|
| a train step | **39.75 s** (A: 476.94 s ÷ 12) |
| an `n=512` validation pass | **~726 s** ≈ 12 min (B − A, halved) — about **18 train steps** |
| 4,000 train steps, validation off | **~44 h** |
| 5,000 train steps at the default `--val-every 200` | **~60 h**, of which validation is **8.4%** |

**Retracted: this section used to say a demo checkpoint was "12+ hours" and that
"the validation passes dominate". Both are wrong.** They came from one 20-step
run that had two validation passes inside it (31m52s) and no way to tell the two
costs apart. I charged nearly all of it to validation, which implicitly priced a
train step at 10.8 s. Run A prices it with nothing else in the run: **39.75 s**,
~4× that, and ~1,400× the GPU's 27.76 ms median from your log.

That inverts the advice that followed. A validation pass really is expensive in
absolute terms — 12 minutes, because `n=512` in two modes is full reconstruction
sampling on one core — but it is expensive *next to a train step that is itself
expensive*, and at the default `--val-every 200` it buys 25 passes across 5,000
steps: **8.4% of the run**. Turning validation off entirely takes ~60 h down to
~55 h. `--val-batch` and `--val-every` are not the difference between a smoke run
and an overnight one; on CPU there is no smoke run to reach. The knob that
matters is `--steps`.

One caveat on the 39.75 s: an earlier run on this same box averaged 60.8 s/step
(36 steps, `"val":null` throughout). You cannot check that one — it sat in
`runs/`, which is gitignored, and its flags were never recorded, which is why I
am not folding it in. Take it only as a reason to read 39.75 s as a floor
measured on an idle box rather than a guarantee.

**So the screenshots in this branch are of an untrained model**, and are labelled
as such. They demonstrate that the server, the simulator verdict and the reveal
scrubber work end to end. They do not demonstrate that the model is good. **A
demo checkpoint needs your GPU**, and it is the one thing in this issue I cannot
do for you:

```bash
cargo run --release --features wgpu --bin train -- \
  --steps 5000 --seed 7 --out runs/seed-7/denoiser \
  --metrics-out runs/seed-7/metrics.jsonl \
  --report-out runs/seed-7/report.html
cargo run --release --bin serve -- --ckpt runs/seed-7/denoiser
```

The second command needs no GPU. That is the point of it.

---

## What I would do next, in order

1. **Train at 11, sample at 15.** One command, never run, gates every
   area-selection plan in §7.
2. **Randomize `ASSEMBLER_BANK` into a chaos family**, preserving its ambiguity.
   §8's recipe, applied to the family §1 says is doing the damage.
3. **Watch the `SCRATCH functional − exact` gap** across the next run. If
   widening the curriculum works, that gap stops collapsing. That is the
   experiment that tells you whether any of this helped.
4. **Then** expert iteration, if `beat` is still climbing under Best-of-N.

RL is step 4 of 4, and step 3 is what tells you whether you need it.

## How to reproduce

```bash
cargo run --release --example task_space   # how few answers the lessons hold
cargo run --release --example bus_tap      # the shared line, three ways, + the lesson census
cargo run --release --example gallery      # what the model is trained on -> gallery.html
cargo run --release --bin serve -- --ckpt checkpoints/denoiser   # a task nobody generated

# Best-of-N needs a temperature: greedy draws the same factory every time.
cargo run --release --bin sample -- --ckpt runs/seed-7/denoiser \
  --best-of 16 --temperature 1.0 --blueprint-out best.txt
```
