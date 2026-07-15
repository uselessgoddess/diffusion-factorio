//! Serve the runtime task editor: paint a factory task in a browser, watch the
//! model design it, read the simulator's verdict on what it drew.
//!
//! `--bin sample` reconstructs *generated* lessons, and `--example gallery`
//! renders a fixed page of them. Both only ever pose tasks the curriculum
//! already knows how to write down. This poses whatever you paint, which is the
//! only way to see whether the model generalizes past its lessons.
//!
//! Usage: `cargo run --release --bin serve -- --ckpt checkpoints/denoiser`
//!
//! Binds loopback by default. It runs a denoising pass for anyone who can reach
//! the port and has no auth, so think before moving it off `127.0.0.1`.

use std::path::PathBuf;

use clap::Parser;
use diffusion_factorio::{persist, serve::serve};

// Inference is always CPU/ndarray: a single denoising run is cheap, and a
// viewer that needs a GPU is a viewer nobody opens.
type B = diffusion_factorio::backend::CpuBackend;

#[derive(Parser)]
#[command(about = "Pose factory tasks by hand and watch a trained denoiser solve them")]
struct Args {
    /// Checkpoint prefix (expects `<ckpt>.mpk` + `<ckpt>.json`).
    #[arg(long, default_value = "checkpoints/denoiser")]
    ckpt: PathBuf,
    /// Address to bind. Loopback by default — see the module docs before
    /// changing it.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Default::default();
    let model = persist::load::<B>(&args.ckpt, &device)?;
    serve(&model, &args.addr, &device)?;
    Ok(())
}
