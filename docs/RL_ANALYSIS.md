# Where the project actually is, and why RL is still not next

Answers [#7](https://github.com/uselessgoddess/diffusion-factorio/issues/7). Every
number here is either transcribed from that issue's own 5,000-step log, or
re-derivable with a command given inline.

---

## TL;DR

**The 5,000-step run is healthy and the model already generates working factories
that nobody baked into it** — at step 5,000, **13.4% of the layouts it builds
from scratch work but are *not* the reference answer**. That is the evidence #7
asks for, and it was already sitting in the log.

**The bad news is in the same number.** That 13.4% is not a plateau, it is a
*decline*: the model was at 67.3% mid-run. Training is actively destroying the
ability the issue wants, because every family in the curriculum has exactly one
labelled answer per task, so **every original solution is a loss penalty**. The
run's last 1,800 steps converted originality into conformity and produced no new
working factories at all.

So the constraint is not the model, the optimizer, or the lack of RL. It is that
**the curriculum has one right answer and the metric only checks the label**.
This branch fixes both, in exactly the order #7 asks for: graded throughput →
Best-of-N verified by the simulator → a curriculum that admits many answers.

**RL is not next.** Its three preconditions are now met — that is *necessary*,
not *sufficient*. The reference already ran this experiment: **PPO, 45M samples,
a full throughput engine — and ≈ 0 on assembler lessons**. It did not deliver.
Meanwhile Best-of-N buys the same thing RL buys for zero training cost and hasn't
been spent; one ambiguous family out of five is a base thin enough that RL would
just memorize "build 3 lines"; and an unverified simulator is a reward function
RL will *exploit* rather than satisfy. When it happens, the first form should be
**expert iteration, not PPO**.

---

## 1. What the 5,000-step run actually says

### 1.1 The eval fix already worked — `1.000` became `0.992` because the eval grew

The previous run reported `exact=1.000` at `n=64`. This one reports `0.992` at
`n=512`. **The model did not get worse; the eval got honest.** For an
all-successes run the 95% lower bound is `0.05^(1/n)`, so 64/64 only ever proved
">95.4%" — the 0.8% tail was always there and the small sample could not contain
it. `val_batch` defaulting 64 → 512 (`src/train.rs:67`) is what surfaced it.

Worth stating plainly because it is the pleasant case: a metric got *worse* and
that was the fix working.

### 1.2 `exact == functional` for 5,000 steps is a property of the *data*

```
VAL n=512 | exact=0.992 functional=0.992
```

These two never separate under `VAL` because **each conditioning has exactly one
valid answer**, so "got it right" and "got it working" are the same event. That
is measurable without a model:

```bash
cargo run --release --example task_space
```
```
MOVE_ONE_ITEM          distinct factories:  41857 | distinct tasks:  41857 | ambiguous tasks: 0
MOVE_ONE_ITEM_CHAOS    distinct factories: 200000 | distinct tasks: 200000 | ambiguous tasks: 0
ASSEMBLER_LINE         distinct factories:    231 | distinct tasks:    231 | ambiguous tasks: 0
UNDERGROUND_CROSS      distinct factories:    110 | distinct tasks:    110 | ambiguous tasks: 0
```

`distinct tasks == distinct factories` in every family: the generator never shows
two answers to one question. A 30-line BFS beats the model at the task as posed.

Two of those families are also just *small*: `assembler_line` has 231 distinct
tasks seen ~173× each in a 5k run, `underground_cross` 110 seen ~364×. That is
memorization scale. `move_one_item` and `..._chaos` are the honest half (~42k and
200k+ tasks, each seen ~once), so `0.992` is real generalization *there* and
recall *elsewhere*.

### 1.3 The model already builds schemes nobody baked in — and training removes them

This is the finding that matters for #7, and it is only visible under `SCRATCH`
(`Sample::blank_to_scaffold` — mask everything but source/sink, so the model must
*design* rather than inpaint). Under `SCRATCH`, `exact` and `functional` **do**
separate. The gap is exactly "layouts that work but are not the reference
answer":

| step | SCRATCH `exact` | SCRATCH `functional` | gap | share of *working* layouts that are original |
|-----:|----------------:|---------------------:|----:|---------------------------------------------:|
|  600 |           0.182 |                0.557 | 0.375 | **67.3%** |
| 3200 |           0.605 |                0.730 | 0.125 | 17.1% |
| 5000 |           0.621 |                0.717 | 0.096 | **13.4%** |

Averaged over checkpoints to keep it out of the noise (n=512, 1 s.e. ≈ ±0.02):

- mid-run (steps 600–2000, 8 checkpoints): mean gap **0.272**
- late (steps 3400–5000, 9 checkpoints): mean gap **0.108**

**The model's originality fell by 2.5× while its ability to build working
factories did not improve at all.** `SCRATCH functional` over steps 3200–5000:
min 0.703, max 0.730, mean 0.717 — a spread of 1.4 s.e., i.e. flat.

The mechanism is not subtle. With one label per task, a working-but-different
layout is indistinguishable from a mistake: the cross-entropy punishes it exactly
as hard. Training converges the model onto the generator's arbitrary tie-break.
**The last 1,800 steps of the run bought conformity, not capability.**

So: the concept works — the model designs original working factories — and the
curriculum is training that out of it.

### 1.4 ~40% of the run was spent not learning

`VAL` saturates around step 2,400–3,000 (0.984 → 0.992 is ~2 s.e. at n=512) but
cosine decay runs to 5,000, ending at `lr 3.08e-11`. `SCRATCH functional` is flat
from 3,200. The gradual-looking tail of the curve is the LR schedule, not
difficulty. Match the schedule to where learning stops and raise batch size —
data generation is only ~7% of a step, so batch 32 simply underutilizes the GPU.

---

## 2. The question #7 really asks

> Я бы хотел уже какие-то реальные полезные схемы иметь возможность генерить
> (которые в неё не закладывали) — чтобы точно убедиться в работоспособности концепции.

**It already does, 13.4% of the time (§1.3), and the old eval could not tell you
whether those schemes were *better* or merely *different*.** That is the actual
gap. `functional` is a yes/no: it says a layout works, never that it works *well*.
Against a rigid curriculum the best achievable `exact` is 1.000 — a model that
perfectly clones the generator is a perfect score. **There is no headroom above
"imitate the BFS" to even measure.**

Which is why the three items #7 lists, in the order it lists them, are the right
ones, and why they had to come before RL:

| # | Ask | Status |
|---|-----|--------|
| 1 | Graded throughput | ✅ `src/throughput.rs` |
| 2 | Best-of-N verified by the simulator | ✅ `src/best_of_n.rs`, `sample --best-of N` |
| 3 | Curriculum that admits many answers | ✅ `ASSEMBLER_BANK` |
| 4 | RL on top of all this | ⛔ **not yet** — §5 |

---

## 3. What this branch adds

### 3.1 Graded throughput (`src/throughput.rs`)

`item_reaches_sink` was binary: it could not rank two working factories, so
Best-of-N had nothing to sort by and RL had no gradient to climb. This was the
blocker for everything downstream.

Now a factory scores in items/second per sink, folded by the reference's power
mean `((1/N)·Σ achievedᵢ^p)^(1/p)` at `p=0.5`, so starving one sink hurts more
than slowing all of them. Flow propagates by Kahn's algorithm over a graph whose
edge `p → q` exists only if `p` pushes into `q` **and** `q` accepts from `p`.

Three deliberate departures from `beyarkay/factorion`, each pinned by a test:

- **The assembler is a real machine.** The reference never reads `crafting_time`
  or `crafting_speed`; it models a machine as a pass-through *ratio* capped at
  1.0, reinterpreting a per-craft count as a per-second rate. That is right for
  0.5 s recipes and 12–20× too generous for long ones, and it means a machine can
  never be the bottleneck — which is exactly what a machine usually *is*. We cap
  at `Recipe::crafts_per_second`.
- **Cycles degrade locally.** The reference scores the whole factory 0 if a cycle
  exists anywhere, even in a disconnected corner — a cliff, as a training signal.
  Here a cycle simply never gets a topological turn, so it starves what is
  downstream of it while sinks fed by other paths still score. Kahn's gives this
  for free; no cycle check needed.
- **No lanes, and the roadmap was wrong to ask for them.** The reference splits
  each belt tile into left/right lane nodes to model sideloading. That is vacuous
  *here*: our entities are 1×1, an inserter has exactly one pickup tile, and belt
  merging already falls out of the per-tile cap. Lanes would be nodes that can
  never differ. The real limitation is the **world model**, not the throughput
  port — so "lane-aware throughput" was struck from the roadmap rather than
  quietly carried forward.

### 3.2 Best-of-N verified by the simulator (`src/best_of_n.rs`)

Draw N layouts, simulate each, keep the winner. **Needs no retraining** — the
sampler is already stochastic via temperature.

```bash
cargo run --release --bin sample -- --ckpt runs/seed-7/denoiser \
  --best-of 16 --temperature 1.0 --blueprint-out best.txt
```

`--best-of N` requires `--temperature > 0`: greedy decoding draws the same
factory every time and the extra passes buy nothing. `BestOfN::distinct` is the
honest probe — **if it stays at 1, the model holds one memorized answer and no
larger N will help** (and neither will a policy gradient, which is why this
number is a precondition for RL rather than a nice-to-have).

### 3.3 A curriculum that admits many answers (`ASSEMBLER_BANK`)

3 sources and a shared sink are the *task*; **how many of the 3 assembler lines
to build is the answer**. All three answers are valid and they are not equally
good:

```bash
cargo run --release --example ambiguity_demo
```
```
=== the task the model is given (the sink asks for CopperCable) ===
......S....
......S....
......S...K
=== 3 valid answers to it ===
-- delivers 0.860 CopperCable/s --   ......SiAiK
-- delivers 1.720 CopperCable/s --   ......SiAiv / ......SiAiK
-- delivers 2.580 CopperCable/s --   ......SiAiv / ......SiAiv / ......SiAiK
Same sources, same sink, same recipe: 3 answers spanning 0.860..2.580/s (3.0x).
Over 20000 seeds, 189 of 189 tasks admit more than one answer.
```

(The rates are the output *inserter* capping at 0.86/s, not the assembler:
CopperCable is 1 plate → 2 cables per 0.5 s, so the machine makes 1.72/s
internally and the inserter is the bottleneck. Each added line adds exactly one
inserter's worth — 1×/2×/3×.)

This is what makes `exact` **wrong** rather than merely hard: it is now capped
below 1.0 by construction, because at most one of the three answers can match the
label and the other two are penalized for being right. It is also what gives
Best-of-N something to choose between and `beat_original` any way to fire.

**A trap worth recording, because it cost a commit.** `Sample::blank` observes
every cell it does not blank, so `removable` must list the region an answer *may*
build, not the cells a given answer *did* build. Listing only the built cells
leaves an unbuilt line observed-as-empty — which states the line count in the
conditioning and silently returns ambiguity to 0. The first version of this
family had that bug, `ambiguity_demo` (which uses `blank_to_scaffold`, observing
only the anchors) happily reported 189/189, and `task_space` (which uses `blank`,
what training actually runs) reported **0**. `task_space` was right. Any new
ambiguous family must be checked under `blank`, not only `blank_to_scaffold` —
the test now checks both.

### 3.4 Wired into the metrics

`thput` (delivered items/s), `ratio` (delivered ÷ the reference answer's rate)
and `beat` (count of layouts that *beat* the reference) now appear in the
progress line, the JSONL and the HTML report — so a run can show throughput
rising while `exact` does not, which is the whole point.

---

## 4. The reference repo, revisited

[`beyarkay/factorion`](https://github.com/beyarkay/factorion) is worth reading and
mostly *not* worth copying wholesale; the long-form comparison is in
[`docs/ANALYSIS.md`](ANALYSIS.md). What changed with this branch:

- **Borrowed and shipped:** the power-mean factory score at exponent `p=0.5`
  (§3.1); real recipe rates in `world.rs`; terminal-only reward as the standing
  decision — they *tried* potential-based shaping and rejected it: −2.8%
  throughput at a p-value of 0.560 (i.e. indistinguishable from noise) for −18.3%
  SPS. A negative result worth inheriting rather than rediscovering.
- **Rejected, with a reason:** lanes (§3.1), the ratio-model assembler (§3.1), the
  global cycle cliff (§3.1).
- **Still worth taking, not yet taken:** their PR #16's 1×1-conv tile head →
  softmax over the flat board (2.6M → **520 params**, no throughput loss, +76.4%
  SPS), and per-tile conditioned attribute heads `P(tile)·P(attrs|tile)`. Cheap,
  and orthogonal to everything above.

**The most useful thing in that repo is a negative result, and it is about RL.**
Their canonical SFT base scores `val/thput_eot ≈ 0.11`, and per-lesson:
`MOVE_ONE_ITEM ≈ 0.38`, **assembler lessons ≈ 0**. With PPO, 45M samples and a
full throughput engine, *they cannot build a working assembler factory from
scratch at all*.

That is worth sitting with before reaching for RL. The reference already ran the
experiment #7 proposes — SOTA-ish RL, real reward, two orders of magnitude more
samples than we have spent — and it did not produce assembler factories. Whatever
is blocking that, a policy gradient did not fix it. This is the single strongest
argument that the next lever is the **task**, not the training algorithm.

**A comparison not to make:** their `0.11` against our `SCRATCH functional=0.717`.
`thput_eot` is a *graded throughput* score; `functional` is *binary* — does it
work, yes or no. Those measure different things, and reading 0.717 as "6× the
reference" would be nonsense. The genuinely comparable number is our new
`SCRATCH ratio` (delivered ÷ the reference answer's rate), which is why that
metric now exists (§3.4). Even then the world models differ (1×1 entities, no
lanes), so it is a sanity check, not a leaderboard. What *is* directly comparable
is the **discipline**: blank everything, rebuild from empty. We adopted that from
them.

---

## 5. RL: preconditions met, still not next

The roadmap gated RL on three things and **all three are now green**: throughput
is graded (§3.1), the sampler can be ranked (§3.2), and at least one family
admits many answers (§3.3). That is necessary, not sufficient.

**Start from the reference's result (§4): PPO + 45M samples + a full throughput
engine ≈ 0 on assembler lessons.** The experiment has been run, at a scale we are
nowhere near, and RL did not deliver the thing we want. That is not proof it
cannot — but it does mean "add RL" is not a plan, and it shifts the burden onto
identifying what actually blocks assembler factories first. Then three concrete
reasons to not do it *next*:

1. **Best-of-N has not been spent.** It buys the same thing RL buys — higher
   delivered throughput — at zero training cost and zero risk of collapse.
   Measure its gain first. It is also the honest read on whether there is a
   distribution to improve: if N draws all land on the same grid
   (`BestOfN::distinct == 1`), a policy gradient has nothing to sharpen either.
2. **One ambiguous family out of five is a thin base.** RL would optimize
   throughput on `ASSEMBLER_BANK` — 189 tasks, seen ~169× each. It could reach a
   perfect score by memorizing "always build 3 lines" without learning one thing
   about design, and the metric would applaud.
3. **The simulator has not been parity-checked against Factorio.** RL optimizes
   the reward it is given, exactly and remorselessly. Hand it an unverified
   simulator and it will find that simulator's bugs rather than good factories —
   the standard failure mode, and far harder to notice than a crash, because the
   reward goes *up*.

**When it does happen, the first form should not be PPO.** Rejection sampling /
expert iteration is the cheapest thing that works: run Best-of-N, keep the
winners, fine-tune on them, repeat. It reuses §3.1 and §3.2 exactly as-is, adds
no new hyperparameters, and cannot collapse the way a policy gradient can. Reward
stays terminal-only.

---

## 6. The playground / RCON alternative

> Либо же это сейчас не главное, а главное сейчас это какой-нибудь интерактивный
> playground или rcon с factorio для runtime inference и проверки схем.

**Not now, but it is on the critical path for RL — and it is precondition #3
above wearing a different hat.**

Blueprint export already exists and is validated end-to-end (PR #4; the
`fbe.teoxoy.com` rejection was fixed in PR #6), so `sample --blueprint-out`
produces a string that pastes into a real game today. An RCON harness would prove
the *simulator* is not lying — which is precisely what RL needs and what nothing
else can supply. It is not CI-able (needs a licensed install), which is why it
sits at step 7 rather than step 1.

An interactive playground is a different thing: it makes the model easier to
*demo*, not easier to *trust*. Given the run in #7 is already at
`SCRATCH functional=0.717`, the bottleneck is what the model is being asked to
learn, not our ability to watch it. Worth building — after the curriculum is
wide enough that watching it is interesting.

---

## 7. Next steps, in order

Full detail and rationale in [`docs/ROADMAP.md`](ROADMAP.md).

1. **Spend Best-of-N** — measure the gain on a real checkpoint; check
   `distinct > 1`. Free throughput, and the go/no-go for RL.
2. **Widen the ambiguous curriculum** — the open half of the work here. Four of
   five families are still rigid; the bank is 189 memorizable tasks. The next
   ambiguous family should be at `move_one_item` scale: multi-source/multi-sink,
   several recipes, tighter obstacle budgets, true 3×3 assemblers and 2×1
   splitters. This is the single highest-value item on the list — §1.3 says the
   curriculum is what is capping the model.
3. **Fix the schedule** — stop at ~3,000 steps or extend the task; raise batch
   size until the GPU saturates (§1.4). ~40% of a run is currently free money.
4. **Cheap architecture wins from the reference** — the 520-param tile head.
5. **Factorio parity via RCON** — earn the right to trust the reward.
6. **Then RL**, as expert iteration first.
