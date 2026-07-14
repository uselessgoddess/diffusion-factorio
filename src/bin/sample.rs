//! Load a checkpoint, blank known factories, and reconstruct them — the
//! always-available "is the model actually building working factories?" check.
//!
//! Prints, per example, the masked input, the model's reconstruction and the
//! ground truth, plus an aggregate reconstruction report (per-channel accuracy,
//! exact-match and — the metric that matters — functional validity).
//!
//! Usage: `cargo run --release --bin sample -- --ckpt checkpoints/denoiser`

use std::path::PathBuf;

use clap::Parser;
use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::metrics::reconstruction_report;
use diffusion_factorio::persist;
use diffusion_factorio::sample::{reconstruct, SampleConfig};
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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Default::default();
    let model = persist::load::<B>(&args.ckpt, &device)?;
    let cfg = SampleConfig {
        steps: args.steps,
        temperature: 0.0,
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

    let recon = reconstruct(&model, &partials, &observed, &cfg, &device);

    for i in 0..args.show.min(recon.len()) {
        println!("=== example {i} [{}] ===", kinds[i].name());
        println!("-- masked input --\n{}", render(&partials[i]));
        println!("-- reconstruction --\n{}", render(&recon[i]));
        println!("-- ground truth --\n{}", render(&originals[i]));
    }

    let report = reconstruction_report(&originals, &recon, &observed);
    println!("\nAGGREGATE: {report}");
    Ok(())
}
