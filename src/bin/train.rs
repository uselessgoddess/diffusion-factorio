//! Train the diffusion denoiser and save a checkpoint.
//!
//! CPU (ndarray) by default; build with `--features wgpu` to train on the GPU:
//!   `cargo run --release --features wgpu --bin train -- --steps 20000`

use std::path::PathBuf;

use clap::Parser;
use diffusion_factorio::diffusion::DiffusionConfig;
use diffusion_factorio::model::DenoiserConfig;
use diffusion_factorio::persist;
use diffusion_factorio::train::{train, TrainConfig};

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
    #[arg(long, default_value_t = 200)]
    val_every: usize,
    #[arg(long, default_value_t = 64)]
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
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Checkpoint path prefix (writes `<out>.mpk` + `<out>.json`).
    #[arg(long, default_value = "checkpoints/denoiser")]
    out: PathBuf,
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
        val_every: args.val_every,
        val_batch: args.val_batch,
        sample_steps: args.sample_steps,
        seed: args.seed,
        model: model_cfg.clone(),
        diffusion: DiffusionConfig::new().with_elbo_weight(args.elbo),
        ..Default::default()
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

    let (model, _logs) = train::<TrainBackend>(&cfg, &device);

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    persist::save(&model, &model_cfg, &args.out)?;
    println!("saved checkpoint to {}.{{mpk,json}}", args.out.display());
    Ok(())
}
