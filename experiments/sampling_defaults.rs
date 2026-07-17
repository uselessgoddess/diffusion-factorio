//! What `steps`, `candidates` and `temperature` are actually worth, measured.
//!
//! The issue asks for defaults with a reason behind them:
//!
//! > не очень понятно как работать со steps, candidates, temperature - нужны
//! > какие-то базовые самые лучшие параметры, чтобы их трогать только для
//! > спецефичных моментов
//!
//! Fair. They were picked by taste. This sweeps one knob at a time against a
//! held-out task set and reports what each setting delivers and what it costs,
//! so the defaults can be argued from a table instead of from taste.
//!
//! The three are not independent, and the cheapest finding is the one the sweep
//! makes unmissable: at `temperature = 0` every candidate is the same greedy
//! draw, so `candidates` multiplies the cost by N and the throughput by exactly
//! nothing. Anything that raises candidates has to raise temperature too.
//!
//! Cost is reported in forward passes (`candidates × steps`) because that is
//! what you pay, linearly, in wall clock.
//!
//! ```text
//! cargo run --release --example sampling_defaults -- <checkpoint> [tasks]
//! ```
//!
//! `<checkpoint>` is a path without extension, as written by [`persist::save`] —
//! the same thing `serve --model` takes. The numbers below describe *that*
//! checkpoint: a weaker model leans on candidates more than a strong one, so
//! re-run this against your own weights rather than inheriting a table trained
//! at someone else's scale.

use burn::prelude::Backend;
use diffusion_factorio::backend::CpuBackend;
use diffusion_factorio::best_of_n::{best_of_n, usable_score, BestOfN, BestOfNConfig};
use diffusion_factorio::factory_gen::{generate, Canvas, LessonKind};
use diffusion_factorio::model::Denoiser;
use diffusion_factorio::persist;
use diffusion_factorio::sample::SampleConfig;
use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::world::Grid;
use std::path::PathBuf;
use std::time::Instant;

/// Grid side. The curriculum only makes square tasks, so this is what the model
/// was trained on.
const SIZE: usize = 11;
/// Seeds start here, well past anything a training run draws, so these tasks are
/// held out rather than remembered.
const SEED_BASE: u64 = 900_000;

/// The baseline every sweep varies one knob away from: today's `serve` defaults.
const BASE_STEPS: usize = 12;
const BASE_CANDIDATES: usize = 8;
const BASE_TEMPERATURE: f64 = 0.9;

/// What a setting delivered on the held-out set.
struct Outcome {
    /// Mean [`usable_score`] of the winning draw — items/s, and zero for a
    /// factory that cannot be built, because nothing is delivered by a factory
    /// nobody can build.
    thput: f64,
    /// Share of winners that export: the ceiling on everything else.
    buildable: f64,
    /// Share of winners where an item reaches a sink at all, at any rate.
    functional: f64,
    /// Mean distinct draws per task. At temperature 0 this is 1.0 by
    /// construction, and it is the number that explains a flat throughput column.
    distinct: f64,
    /// Forward passes per task: what the setting costs.
    passes: usize,
    seconds: f64,
}

fn measure<B: Backend>(
    model: &Denoiser<B>,
    partials: &[Grid],
    observed: &[Vec<bool>],
    steps: usize,
    candidates: usize,
    temperature: f64,
    device: &B::Device,
) -> Outcome {
    let cfg = BestOfNConfig {
        n: candidates,
        sample: SampleConfig {
            steps,
            temperature,
            seed: 7,
        },
        prefer_compact: true,
    };
    let started = Instant::now();
    let runs = best_of_n(model, partials, observed, &cfg, device);
    let seconds = started.elapsed().as_secs_f64();

    let tasks = runs.len() as f64;
    let mean = |f: &dyn Fn(&BestOfN) -> f64| runs.iter().map(f).sum::<f64>() / tasks;
    Outcome {
        thput: mean(&|r| usable_score(&r.best.grid)),
        buildable: mean(&|r| f64::from(r.best.grid.is_consistent())),
        functional: mean(&|r| f64::from(item_reaches_sink(&r.best.grid))),
        distinct: mean(&|r| r.distinct as f64),
        passes: steps * candidates,
        seconds,
    }
}

fn header(knob: &str) {
    println!("\n{knob:>12}  items/s  buildable  functional  distinct  passes  wall");
}

fn row(label: String, o: &Outcome) {
    println!(
        "{label:>12}  {:>7.3}  {:>8.0}%  {:>9.0}%  {:>8.2}  {:>6}  {:>4.1}s",
        o.thput,
        100.0 * o.buildable,
        100.0 * o.functional,
        o.distinct,
        o.passes,
        o.seconds,
    );
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let checkpoint = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("usage: sampling_defaults <checkpoint> [tasks]"))?,
    );
    let tasks: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(16);

    let device = Default::default();
    let model = persist::load::<CpuBackend>(&checkpoint, &device)?;

    // Held-out tasks, spread across every lesson family so no single one drives
    // the table.
    let (mut partials, mut observed) = (Vec::new(), Vec::new());
    for i in 0..tasks {
        let kind = LessonKind::all()[i % LessonKind::all().len()];
        let Some(sample) = generate(kind, Canvas::square(SIZE), SEED_BASE + i as u64) else {
            continue;
        };
        // Only the sources and sinks survive: the model gets the task, not a
        // head start on the answer. This is what `serve` hands it.
        let (partial, obs) = sample.blank_to_scaffold();
        partials.push(partial);
        observed.push(obs);
    }
    println!(
        "{} held-out tasks on {SIZE}x{SIZE}, ports given and nothing else.\n\
         Checkpoint: {}",
        partials.len(),
        checkpoint.display()
    );
    println!(
        "Baseline: steps={BASE_STEPS} candidates={BASE_CANDIDATES} temperature={BASE_TEMPERATURE}"
    );

    let run = |steps, candidates, temperature| {
        measure(
            &model,
            &partials,
            &observed,
            steps,
            candidates,
            temperature,
            &device,
        )
    };

    header("steps");
    for steps in [4, 8, 12, 16, 24, 32] {
        let o = run(steps, BASE_CANDIDATES, BASE_TEMPERATURE);
        row(format!("{steps}"), &o);
    }

    header("candidates");
    for candidates in [1, 2, 4, 8, 16, 32] {
        let o = run(BASE_STEPS, candidates, BASE_TEMPERATURE);
        row(format!("{candidates}"), &o);
    }

    header("temperature");
    for temperature in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5] {
        let o = run(BASE_STEPS, BASE_CANDIDATES, temperature);
        row(format!("{temperature:.1}"), &o);
    }

    // The interaction that matters: candidates only buy anything if the draws
    // differ, and only temperature makes them differ. A column that stays flat
    // across candidates at temperature 0 is not a weak model, it is the same
    // factory drawn N times.
    header("t=0, cands");
    for candidates in [1, 8, 32] {
        let o = run(BASE_STEPS, candidates, 0.0);
        row(format!("{candidates}"), &o);
    }

    println!(
        "\n`distinct` is the column to read first: it counts how many different\n\
         factories the N draws actually produced. Where it sits at 1.00, every\n\
         extra candidate is a forward pass spent redrawing the same answer."
    );
    Ok(())
}
