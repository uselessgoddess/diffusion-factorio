//! Load a checkpoint, blank known factories, and reconstruct them — the
//! always-available "is the model actually building working factories?" check.
//!
//! Prints, per example, the masked input, the model's reconstruction and the
//! ground truth, plus an aggregate reconstruction report (per-channel accuracy,
//! exact-match, functional validity and delivered throughput).
//!
//! Usage: `cargo run --release --bin sample -- --ckpt checkpoints/denoiser`
//!
//! With `--best-of N --temperature T` it draws `N` candidates per task and keeps
//! whichever one the simulator scores highest — including for `--blueprint-out`,
//! so the exported blueprint is the best factory the model could find rather
//! than the first one it happened to produce.

use std::path::PathBuf;

use clap::Parser;
use diffusion_factorio::best_of_n::{best_of_n, BestOfN, BestOfNConfig};
use diffusion_factorio::blueprint::{blueprint_string, grid_to_blueprint};
use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::metrics::reconstruction_report;
use diffusion_factorio::observability::{write_sample_report, SampleReportEntry};
use diffusion_factorio::persist;
use diffusion_factorio::sample::{reconstruct_with_diagnostics, SampleConfig};
use diffusion_factorio::textual::render;
use diffusion_factorio::world::Grid;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

// Inference is always CPU/ndarray (cheap; no autodiff needed).
type B = diffusion_factorio::backend::CpuBackend;

#[derive(Parser)]
#[command(about = "Reconstruct blanked factories with a trained denoiser")]
struct Args {
    /// Checkpoint prefix (expects `<ckpt>.mpk` + `<ckpt>.json`).
    #[arg(long, default_value = "checkpoints/denoiser")]
    ckpt: PathBuf,
    #[arg(long, default_value_t = 11)]
    size: usize,
    /// How many examples to print in detail.
    #[arg(long, default_value_t = 4)]
    show: usize,
    /// Aggregate report over this many blanked factories.
    #[arg(long, default_value_t = 128)]
    eval: usize,
    #[arg(long, default_value_t = 12)]
    steps: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Softmax temperature. `0` is greedy/deterministic; above zero the sampler
    /// draws, which is what gives `--best-of` something to choose between.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    /// Draw this many candidates per task and keep the one the simulator scores
    /// highest. Costs one extra sampling pass per candidate and no retraining.
    #[arg(long, default_value_t = 1)]
    best_of: usize,
    /// Offline spatial confidence/entropy/error heatmap report.
    #[arg(long, default_value = "sample-report.html")]
    report: PathBuf,
    /// Export the first reconstruction as an importable Factorio blueprint.
    #[arg(long)]
    blueprint_out: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.eval > 0, "--eval must be at least 1");
    anyhow::ensure!(args.best_of > 0, "--best-of must be at least 1");
    anyhow::ensure!(
        args.best_of == 1 || args.temperature > 0.0,
        "--best-of {} needs --temperature above 0: greedy decoding draws the same \
         factory every time, so the extra passes would cost compute and change nothing",
        args.best_of,
    );
    let device = Default::default();
    let model = persist::load::<B>(&args.ckpt, &device)?;
    let cfg = SampleConfig {
        steps: args.steps,
        temperature: args.temperature,
        seed: args.seed,
    };

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(0xB1A2));

    // Build an evaluation set of blanked factories across all feasible lessons.
    let mut originals: Vec<Grid> = Vec::new();
    let mut partials: Vec<Grid> = Vec::new();
    let mut observed: Vec<Vec<bool>> = Vec::new();
    let mut kinds: Vec<LessonKind> = Vec::new();
    let mut ctr = args.seed;
    while originals.len() < args.eval {
        let kind = LessonKind::all()[originals.len() % LessonKind::all().len()];
        ctr += 1;
        if let Some(s) = generate(kind, args.size, ctr) {
            let (partial, obs) = s.blank(None, &mut rng);
            originals.push(s.solution);
            partials.push(partial);
            observed.push(obs);
            kinds.push(kind);
        }
    }

    let picks = (args.best_of > 1).then(|| {
        best_of_n(
            &model,
            &partials,
            &observed,
            &BestOfNConfig {
                n: args.best_of,
                sample: cfg.clone(),
            },
            &device,
        )
    });
    let diagnostics = match &picks {
        Some(picks) => picks.iter().map(|pick| pick.best.clone()).collect(),
        None => reconstruct_with_diagnostics(&model, &partials, &observed, &cfg, &device),
    };
    let recon: Vec<Grid> = diagnostics
        .iter()
        .map(|result| result.grid.clone())
        .collect();

    for i in 0..args.show.min(recon.len()) {
        println!("=== example {i} [{}] ===", kinds[i].name());
        println!("-- masked input --\n{}", render(&partials[i]));
        println!("-- reconstruction --\n{}", render(&recon[i]));
        println!("-- ground truth --\n{}", render(&originals[i]));
    }

    let report = reconstruction_report(&originals, &recon, &observed);
    println!("\nAGGREGATE: {report}");

    if let Some(picks) = &picks {
        let mean = |f: fn(&BestOfN) -> f64| picks.iter().map(f).sum::<f64>() / picks.len() as f64;
        let (first, best) = (mean(BestOfN::first_score), mean(BestOfN::best_score));
        let distinct = mean(|pick| pick.distinct as f64);
        println!(
            "BEST-OF-{}: one draw delivers {first:.3}/s, best of {} delivers {best:.3}/s \
             (+{:.1}%) | {distinct:.2} distinct factories per task",
            args.best_of,
            args.best_of,
            if first > 0.0 {
                100.0 * (best - first) / first
            } else {
                0.0
            },
        );
        if distinct < 1.5 {
            println!(
                "  note: the draws barely differ, so Best-of-N has little to choose between. \
                 Either raise --temperature, or the model has collapsed to one answer per task."
            );
        }
    }

    let entries: Vec<SampleReportEntry<'_>> = (0..args.show.min(recon.len()))
        .map(|i| SampleReportEntry {
            label: kinds[i].name(),
            input: &partials[i],
            prediction: &diagnostics[i].grid,
            target: &originals[i],
            observed: &observed[i],
            confidence: &diagnostics[i].confidence,
            entropy: &diagnostics[i].entropy,
            reveal_step: &diagnostics[i].reveal_step,
        })
        .collect();
    write_sample_report(&args.report, &entries)?;
    println!("saved spatial diagnostics to {}", args.report.display());

    if let Some(path) = &args.blueprint_out {
        let blueprint = grid_to_blueprint(&recon[0], "diffusion-factorio reconstruction")?;
        let encoded = blueprint_string(&blueprint)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, encoded)?;
        println!("saved Factorio blueprint to {}", path.display());
    }
    Ok(())
}
