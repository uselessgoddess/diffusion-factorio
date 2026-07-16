# Why the 10,000-step run does not generalize

Issue #13 trains for 10,000 steps on 11×11, runs `serve` on 13×9, and gets
factories that deliver nothing on tasks a human solves by eye. This document
answers why, in the order the issue asks.

Every claim here is either a quoted line of code or a table some
`experiments/*.rs` prints. Where a number comes from a checkpoint, the
checkpoint is named, because one of them is far too small to trust and it
matters which claims rest on it.

## TL;DR

The final log is not the log of a model that failed to learn:

```
step 10000/10000 | loss 0.1079 | place 0.99 | acc[E=0.98 D=0.98 I=1.00 M=1.00]
  VAL     n=512 | exact=0.646 functional=0.912 consistent=0.869 | thput=6.034/s
  SCRATCH n=512 | exact=0.391 functional=0.736 consistent=0.748 | thput=5.694/s
```

It is the log of a model that learned what it was scored on. Five separate
things sat between that log and a factory that runs, and none of them were
the model's weights:

1. **The decoder could not emit a legal cell even when the model knew the
   answer.** Four channels were argmaxed independently. Fixed — `c1f98eb`.
2. **Best-of-N ranked unbuildable draws above buildable ones.** Fixed — `053b439`.
3. **The ASCII view drew the violations as clean floor**, so none of this was
   visible. Fixed — `4b23104`.
4. **A machine with no recipe was a legal cell**, so a dead factory scored
   `consistent=1.0`. Fixed — `03e3341`.
5. **The curriculum is square-only** and the issue infers on 13×9. Not fixed;
   measured below, and it is the one that needs a real decision.

Faults 1 and 4 are both visible in the issue's own two examples, and fault 5 is
the reason the issue's framing — «независимо от места на grid» — is the right
framing.

## The first example, answered

The issue asks a direct question about its first result:

> Хотя вроде как тут всё окей, разве что два конвейера входят в sink. Или я тут
> не прав?

**The two belts are innocent, and the layout is not окей.** A sink accepts from
anything (`throughput.rs`, `accepts_from`: `Entity::Sink => true`), and the
chain in the corner is fed by nothing, so it carries nothing into the sink.

`experiments/issue_examples.rs` rebuilds the layout from the blueprint string in
the issue — not from the ASCII, which cannot show an inserter's facing and so
cannot settle the question — and repairs it one fault at a time:

```
                                           items/s  legal   sink  export
  as the model built it                    0.000/s     NO     no      NO
  without the two-belts-into-sink chain    0.000/s     NO     no      NO
  + assembler given the gear recipe        0.000/s    yes    yes     yes
  + input inserter instead                 0.000/s     NO     no      NO
  + both repairs                           0.430/s    yes    yes     yes
```

Deleting the suspected chain changes the score by exactly 0.000. The real faults
are two, and neither repair helps alone:

* **The assembler carries no recipe.** The exported `assembling-machine-1` has no
  `recipe` field, and `blueprint.rs` writes one whenever the cell has a craftable
  tag, so its absence proves the cell had none. `sim::emits` crafts nothing
  without a recipe.
* **Six belts carry iron plate into the assembler's wall.** Belts do not load
  machines — `throughput.rs`, `accepts_from`: only `Entity::Inserter |
  Entity::Source` may feed an `Entity::Assembler`. The model built the *output*
  inserter correctly and omitted the input one.

`0.430/s` is `INSERTER_RATE 0.86 ÷ 2 plates per gear`, exactly: even repaired,
the single input inserter is the bottleneck.

### «она почему-то стремится производить тут gear, хотя я не просил»

It was asked. The decoded sink at `12.5 4.5` is tagged `iron-gear-wheel`. Gears
exist only as an assembler recipe, so building an assembler is the only response
to that task that could ever score. **The model was right about the plan and
wrong about the details** — which is the encouraging half of this issue.

### The second example is one cell from working

It *has* the input inserter at (6,4). Its only fault is the `Direction::None`
belt at (11,4) — the mode-splitting bug fixed in `c1f98eb`. That example is a
direct validation of that fix.

On the obstacle in that example: reading the ASCII, the obstacles sit at (6,3)
and (6,5) and the inserter at (6,4) is in the free gap, so that layout did not
route through an obstacle — and `Grid::footprints_are_legal()` would reject one
that did. The suspicion was reasonable, though, because the renderer could not
have shown such a violation. It can now (`4b23104`): `!` is built-on-obstacle.

## Root cause 1: the decoder invented cells the model never proposed

This is the `Not importable: cannot export inconsistent cell at (11, 4):
Cell { entity: TransportBelt, direction: None, ... }` in the issue's log, and it
was not a model error at all.

The four channels were decoded independently, each by its own argmax. So when
the entity head is confident *something* is there and the direction head is
split between two ways it could face, per-channel argmax takes the majority from
one and the plurality from the other and emits the combination — `TransportBelt +
Direction::None` — that **neither head ever proposed as a whole cell**. The
model can be right and the decoder still wrong.

`decode_cell` now scores the 45 legal combinations jointly and picks the
likeliest whole cell. An illegal cell is no longer representable, so that export
error cannot recur by this route.

## Root cause 4: a machine with no recipe was legal

`Cell::is_consistent` asked only that an item-bearing entity be *allowed* an item
tag, never that it carry one. So `Assembler + Item::None` passed every check:

* `sim::emits` returns nothing for it — a nine-tile machine that cannot craft.
* `blueprint.rs` exported it as an `assembling-machine-1` with no recipe, one you
  would have to click a recipe into by hand.
* The `consistent` metric gave it full marks.
* `sample::decode_cell` draws only from the legal table, so the decoder was free
  to propose it — and did, in the issue's first example.

An assembler must now carry a tag naming a real recipe. The legal table drops
from 57 rows to 45, all of it assembler (24 → 12). **No training target moves**:
every lesson in `factory_gen` takes its recipe from `single_input_craftable()` or
`craftable()`, so none of them ever built the cell this removes.

### Why `acc[item=1.00]` never caught it

Because ~95% of cells are empty, the item channel is ~99% `Item::None`. A model
that predicts `None` everywhere scores near 1.00. The one cell where the item
channel carries the entire task — the assembler's recipe — is invisible in that
average. **That is how a recipe-less assembler survived 10,000 steps at
`I=1.00`.** Per-channel accuracy is a training diagnostic, not a quality metric.

## The metrics disagree with the simulator

Row 3 of the ablation above says `sink=yes` while the layout delivers `0.000/s`.
That is not a contradiction, it is a gap between two metrics:

* `item_reaches_sink` "accepts from any pusher" **on purpose** (`sim.rs:57`) and
  does not model the belts-do-not-load-machines rule.
* `throughput::score` does.

So **`functional=0.912` counts layouts that deliver nothing** — specifically, the
exact fault in the issue's first example. `functional` is a reachability check,
not a verdict. `thput` is the honest column.

A second gap, worth recording: `throughput::score` still returns positive for
grids that cannot be built (two assemblers on one tile simulate fine). `053b439`
scoped the fix to `best_of_n::usable_score` rather than to `throughput::score`
itself, because that function also feeds `thput`/`ratio`/`beat` and would be the
RL reward channel — a wider blast radius than this issue should take on.

## «Модель генерирует всю схему за последние 2-3 шага»

The issue's two observations here — that the animation resolves at the end, and
that raising steps to 24 helps — are one fact, and it is the schedule's, not the
model's.

`sample::still_masked_after` reveals on a cosine: cumulative revealed after round
k is `1 − cos(π/2 · (k+1)/steps)`. That curve is flat early and steep late, so
**most rounds pass before most cells are decided at every setting** — cosine
crosses 0.5 at exactly 2/3 of the way through, so ~70% of rounds run before half
the grid is committed, whatever `steps` is.

`experiments/reveal_schedule.rs` measures it on a 13×9 grid:

| steps | committed in last 3 rounds | committed in the final round |
|------:|---------------------------:|-----------------------------:|
| 12    | 38.3%                      | 13.0%                        |
| 24    | 19.1%                      | 7.0%                         |

So `steps` does not change *when* the model commits. It changes **how big the
final irreversible chunk is** — and that is why 24 looked better. The animation
is honest; it is drawing what the schedule does.

## steps, candidates, temperature

> нужны какие-то базовые самые лучшие параметры ... непонятно они по-приколу
> взяты или дефолтные варианты лучшие

They are the defaults: `serve` uses `candidates=8, steps=12, temperature=0.9`
(`src/serve.rs`). `experiments/sampling_defaults.rs` measures what each is worth.

**Caveat, and it is a big one:** the table below is from a 1,200-step,
`hidden=32` CPU checkpoint — far weaker than the issue's model (`exact=0.125`
against its `0.646`). Read the *shape* of the tradeoff, not the absolute
numbers, and re-run it on the real weights: the experiment takes a checkpoint
path precisely so that is one command.

```
       steps  items/s  buildable  functional  distinct  passes  wall
           4    0.938        81%          6%      7.56      32   5.1s
           8    1.902        88%         19%      7.06      64   9.7s
          12    1.018        88%         19%      6.44      96  15.0s
          24    1.929        94%         25%      5.25     192  29.4s
          32    0.938        81%         12%      5.81     256  38.5s

  candidates  items/s  buildable  functional  distinct  passes  wall
           1    0.991        69%         12%      1.00      12   1.7s
           8    1.018        88%         19%      6.44      96  14.0s
          32    1.929        94%         25%     21.19     384  62.5s

 temperature  items/s  buildable  functional  distinct  passes  wall
         0.0    0.107        75%         12%      1.00      96  13.0s
         0.9    1.018        88%         19%      6.44      96  14.6s
         1.5    0.991        94%         12%      7.69      96  14.8s

  t=0, cands  items/s  buildable  functional  distinct  passes  wall
           1    0.107        75%         12%      1.00      12   1.7s
          32    0.107        75%         12%      1.00     384  62.0s
```

What survives the caveat, because it is a property of the algorithm rather than
of the weights:

* **`temperature=0` and `candidates>1` is always waste.** At zero the draw is a
  deterministic argmax, so all N candidates are byte-identical: the last block
  spends 384 forward passes to redraw one answer 32 times, and scores exactly
  what 12 passes scored. `distinct` is the column that shows it. `bin/sample.rs`
  already asserts against this combination; `serve` defaults to 0.9 and avoids
  it.
* **Cost is exactly linear** in `steps × candidates`, and both buy *buildability*
  rather than throughput: 69% → 94% across the candidates sweep.
* **`items/s` is noise here** at n=16 tasks — it is not monotone in anything.
  Do not read a ranking into it.

The measured tradeoff agrees with the issue's own finding that 24 steps beats 12,
which is the one thing in this table confirmed independently of the checkpoint.

**Recommended defaults:** keep `candidates=8, temperature=0.9`, raise `steps` to
16–24. Turn `candidates` up when you want a *buildable* answer and have passes to
spend; leave `temperature` alone unless `distinct` is near 1 (raise it) or the
draws are incoherent (lower it). None of these fix a task the model cannot do —
they are worth a few points of buildability, not a new capability.

## Root cause 5: the model has never seen the canvas it was asked about

This is the issue's own framing, and it is measurable:

> Максимально обобщить, чтобы модель понимала что для производства шестерёнок
> нужно пластины железные в сборщик любой ценой запихнуть и как-то уместить,
> **независимо от места на grid**.

Every lesson in `factory_gen` builds on `Grid::new(size, size)`. **A square
canvas is the only shape the model has ever been shown.** The issue trained on
11×11 and inferred on 13×9.

`model::tests::one_set_of_weights_runs_at_any_grid_size` proves the weights *run*
at any size — the denoiser is fully convolutional, so nothing throws. It does not
prove they *work*, and those are different claims.

`experiments/grid_shape.rs` holds the task identical — iron plate from a source
to a sink exactly 6 cells east, both centred — and changes only the canvas around
it, so a longer route cannot be the confound:

```
      canvas  cells  items/s  buildable  functional  note
    11x11      121    3.118       100%         64%  trained shape
    13x9       117    0.478       100%         56%  the issue's shape
     9x13      117    1.551       100%         54%  the same, turned
    11x9        99    0.573       100%         67%
     9x11       99    0.547       100%         64%
    13x13      169    3.858       100%         69%
    15x15      225    1.516       100%         67%
     9x9        81    3.333       100%         22%
```

11×11 and 13×9 are within four cells of the same area and hold the identical
task. **The trained shape scores 6.5× the issue's shape.** Every square canvas
in the trained size range (9×9, 11×11, 13×13) scores above every rectangular one;
`buildable` is 100% throughout, so the model is not drawing garbage — it is
drawing legal factories that do not deliver.

Same caveats as above — weak checkpoint, 9–15 tasks per row, one seed — so treat
this as a strong hypothesis with a reproduction command attached rather than a
settled number. It is the single most valuable thing to re-run on the real
weights, because if it holds there, **it means most of the issue's inference runs
were posed on a distribution the model was never trained on**, and no amount of
`steps` or `candidates` repairs that.

### What to do about it

Padding a square lesson into a rectangular canvas at a random offset is the cheap
version: valid targets, no generator rewrite, and it teaches "the canvas shape is
not the task" plus translation diversity. It does *not* teach the model to route
through the extra space.

The honest version is to thread `(width, height)` through `factory_gen` instead of
one `size`, and sample a shape per batch (a batch must share a tensor shape, so
per-batch is the granularity). That touches every generator and their
`random_cell`/`neighbours`/`step` helpers. It is the change this issue actually
argues for, and it is worth doing properly rather than approximating.

## What the reference repo achieved, and at what cost

> Посмотри как в оригинальном репозитории было достигнуто большее качество и было
> ли достигнуто. И каким путём, какой ценой.

**It was not achieved.** On the full curriculum, [factorion][ref] reports
`val/thput_eot ≈ 0.11`, and its assembler lessons sit near zero — the same wall
this repo is at, from the other side.

* The ~0.95 number is **one isolated lesson** (PR #290, merged 2026-07-13), not
  the curriculum. Its own writeup calls the p-values "soft" and it is n=2
  effective.
* The change that produced it was **mean+max global-context pooling** — the only
  measured win in their history, and the one thing here worth borrowing.
* **Cost:** ≈12.5 GPU-hours per 45M-sample SFT run on an RTX 2000 Ada.
* Their PR #16's "2.6M → 520 params" is only the *action pathway*; throughput
  moved +3.3% at p=0.701 — not significant. The real win there was +76.4% SPS.
* They decode **per-channel argmax with no joint decoding**, so root cause 1 is
  ours to fix, not theirs to lend.
* They are square-only and size-locked, so they cannot answer the shape question.

Their issue #263 concedes the model "memorises recipe priors and never learns the
general rule", and that conv stacks are "the wrong operation class" for the
variable-offset gather that reading a sink's item tag requires — **precisely the
generalization this issue is asking for.** The reference is evidence that more of
the same training does not get there.

One correction while we are here: their FOOTPRINT leak is fixed upstream, so
`docs/ANALYSIS.md`'s entry on it was reading as a live criticism of a bug that no
longer exists. It now says so.

[ref]: https://github.com/beyarkay/factorion/

## What this means for «инновационный подход»

> Если дело в датасете или методе обучения - двигайся в этом направлении.

The evidence points at the training method, which matches the issue's own
instinct — but not at a bigger model. Three of the five faults were in the
decoder and the metrics, and the model was carrying more signal than the pipeline
could express. Before changing the architecture:

1. **Re-run the shape experiment on the real weights.** If 13×9 really is 6.5×
   worse than 11×11, that is the whole ballgame and the fix is curriculum, not
   capacity.
2. **Score what you want.** `functional` counts layouts that deliver nothing and
   `acc[item]` is 99% `None`; a model optimized against those is optimized
   against the wrong thing. `thput` and `usable_score` are the honest channels.
3. **Then** consider mean+max global pooling — the reference's one measured win,
   and cheap.

RL against `usable_score` is the natural next lever, and `docs/RL_ANALYSIS.md`
already works through it. It is worth noting that RL would have been *actively
harmed* by faults 2 and 4: an agent rewarded on a channel that scores unbuildable
grids and dead machines learns to build them. Fixing the reward channel was a
prerequisite, and now it is done.

## Honest summary

Nothing here required the model to be smarter. Four of five faults were fixed by
reading the code and the issue's own blueprint strings, and each one is now a
test that fails without the fix. The fifth — the square-only curriculum — is the
one that needs a decision, and the one the issue was right about from the start.
