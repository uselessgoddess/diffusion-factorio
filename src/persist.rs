//! Save / load a trained denoiser and its config.
//!
//! We persist two things side by side: the model weights (`<path>.mpk` via burn's
//! `CompactRecorder`) and the [`DenoiserConfig`] as JSON (`<path>.json`), so a
//! checkpoint is self-describing — `load` re-instantiates the exact architecture
//! before loading weights.

use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use std::path::Path;

use crate::model::{Denoiser, DenoiserConfig};

/// Save `model` (weights) and `config` (architecture) under `path` (no ext).
pub fn save<B: Backend>(
    model: &Denoiser<B>,
    config: &DenoiserConfig,
    path: &Path,
) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(path.with_extension("json"), json)?;
    model
        .clone()
        .save_file(path.to_path_buf(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("save weights: {e}"))?;
    Ok(())
}

/// Load a denoiser previously written by [`save`].
pub fn load<B: Backend>(path: &Path, device: &B::Device) -> anyhow::Result<Denoiser<B>> {
    let json = std::fs::read_to_string(path.with_extension("json"))?;
    let config: DenoiserConfig = serde_json::from_str(&json)?;
    let model = config.init::<B>(device);
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), device)
        .map_err(|e| anyhow::anyhow!("load weights: {e}"))?;
    Ok(model.load_record(record))
}
