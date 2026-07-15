//! Train the diffusion denoiser and save a checkpoint.
//!
//! CPU (ndarray) by default; build with `--features wgpu` to train on the GPU:
//!   `cargo run --release --features wgpu --bin train -- --steps 20000`

use std::path::PathBuf;

use clap::Parser;
use diffusion_factorio::diffusion::DiffusionConfig;
use diffusion_factorio::model::DenoiserConfig;
use diffusion_factorio::observability::{write_training_report, MetricsWriter, RunMetadata};
use diffusion_factorio::persist;
use diffusion_factorio::train::{train_with_observer, TrainConfig};

#[cfg(feature = "wgpu")]
type TrainBackend = diffusion_factorio::backend::GpuAutodiff;
#[cfg(not(feature = "wgpu"))]
type TrainBackend = diffusion_factorio::backend::CpuAutodiff;

#[derive(Parser)]
#[command(about = "Train the masked-diffusion factory denoiser")]
struct Args {
    #[arg(long, default_value_t = 11)]
    size: usize,
    #[arg(long, default_value_t = 5000)]
    steps: usize,
    #[arg(long, default_value_t = 32)]
    batch: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 100)]
    warmup: usize,
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,
    #[arg(long, default_value_t = 200)]
    val_every: usize,
    /// Held-out factories per validation pass. 64 cannot distinguish a perfect
    /// model from an 83%-per-lesson one; see docs/TRAINING_ANALYSIS.md.
    #[arg(long, default_value_t = 512)]
    val_batch: usize,
    #[arg(long, default_value_t = 12)]
    sample_steps: usize,
    #[arg(long, default_value_t = 64)]
    hidden: usize,
    #[arg(long, default_value_t = 6)]
    blocks: usize,
    /// Use the MDLM continuous-time ELBO (1/t) weighting.
    #[arg(long, default_value_t = false)]
    elbo: bool,
    /// Minimum sampled diffusion time (bounds ELBO variance).
    #[arg(long, default_value_t = 0.02)]
    t_min: f64,
    /// Extra loss weight for non-empty cells (prevents empty collapse).
    #[arg(long, default_value_t = 8.0)]
    structure_weight: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Checkpoint path prefix (writes `<out>.mpk` + `<out>.json`).
    #[arg(long, default_value = "checkpoints/denoiser")]
    out: PathBuf,
    /// Durable one-record-per-step metrics (safe if a run is interrupted).
    #[arg(long, default_value = "runs/training-metrics.jsonl")]
    metrics_out: PathBuf,
    /// Self-contained offline HTML with curves and parameter explanations.
    #[arg(long, default_value = "runs/training-report.html")]
    report_out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device: burn::tensor::Device<TrainBackend> = Default::default();

    let model_cfg = DenoiserConfig::new()
        .with_hidden(args.hidden)
        .with_blocks(args.blocks);
    let cfg = TrainConfig {
        grid_size: args.size,
        steps: args.steps,
        batch_size: args.batch,
        lr: args.lr,
        warmup: args.warmup,
        grad_clip: args.grad_clip,
        val_every: args.val_every,
        val_batch: args.val_batch,
        sample_steps: args.sample_steps,
        seed: args.seed,
        model: model_cfg.clone(),
        diffusion: DiffusionConfig::new()
            .with_elbo_weight(args.elbo)
            .with_t_min(args.t_min)
            .with_structure_weight(args.structure_weight),
    };

    println!(
        "backend: {}",
        if cfg!(feature = "wgpu") {
            "wgpu (GPU)"
        } else {
            "ndarray (CPU)"
        }
    );
    println!(
        "training {} steps on {}x{} grids...",
        cfg.steps, cfg.grid_size, cfg.grid_size
    );

    let mut metrics_writer = MetricsWriter::create(&args.metrics_out)?;
    let mut metrics_error = None;
    let (model, logs) = train_with_observer::<TrainBackend, _>(&cfg, &device, |log| {
        if metrics_error.is_none() {
            metrics_error = metrics_writer.append(log).err();
        }
    });
    if let Some(error) = metrics_error {
        return Err(error);
    }

    let metadata = RunMetadata {
        backend: if cfg!(feature = "wgpu") {
            "wgpu (GPU)".to_owned()
        } else {
            "ndarray (CPU)".to_owned()
        },
        grid_size: cfg.grid_size,
        steps: cfg.steps,
        batch_size: cfg.batch_size,
        val_batch: cfg.val_batch,
        sample_steps: cfg.sample_steps,
        seed: cfg.seed,
        peak_lr: cfg.lr,
        warmup_steps: cfg.warmup,
        grad_clip: cfg.grad_clip,
        hidden: cfg.model.hidden,
        blocks: cfg.model.blocks,
        embed_dim: cfg.model.embed_dim,
        time_dim: cfg.model.time_dim,
        elbo_weight: cfg.diffusion.elbo_weight,
        t_min: cfg.diffusion.t_min,
        structure_weight: cfg.diffusion.structure_weight,
    };
    write_training_report(&args.report_out, &metadata, &logs)?;

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    persist::save(&model, &model_cfg, &args.out)?;
    println!("saved checkpoint to {}.{{mpk,json}}", args.out.display());
    println!("saved metrics to {}", args.metrics_out.display());
    println!("saved training report to {}", args.report_out.display());
    Ok(())
}
