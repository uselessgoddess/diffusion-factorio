//! Manual burn training loop for the masked-diffusion denoiser.
//!
//! Deliberately explicit (no `Learner` abstraction) so every moving part the
//! user cares about is visible: the streaming lesson generator, the AdamW step,
//! the warmup+cosine LR schedule, gradient clipping, and — crucially — the
//! periodic *functional* validation that proves the model is actually learning
//! to build working factories, not just lowering a loss number.

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

use crate::data::GridBatch;
use crate::diffusion::{loss, DiffusionConfig};
use crate::factory_gen::{generate, LessonKind, Sample};
use crate::metrics::{reconstruction_report, ReconReport};
use crate::model::{Denoiser, DenoiserConfig};
use crate::sample::{reconstruct, SampleConfig};
use crate::world::Grid;

/// Everything needed to run a training session.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub grid_size: usize,
    /// Number of optimizer steps.
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Linear warmup steps before cosine decay.
    pub warmup: usize,
    pub grad_clip: f32,
    /// Run validation (and log) every this many steps.
    pub val_every: usize,
    pub val_batch: usize,
    /// Reverse-diffusion rounds used during validation.
    pub sample_steps: usize,
    pub seed: u64,
    pub model: DenoiserConfig,
    pub diffusion: DiffusionConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            grid_size: 11,
            steps: 2000,
            batch_size: 32,
            lr: 3e-4,
            warmup: 100,
            grad_clip: 1.0,
            val_every: 100,
            val_batch: 64,
            sample_steps: 12,
            seed: 0,
            model: DenoiserConfig::new(),
            diffusion: DiffusionConfig::new(),
        }
    }
}

/// One line of training telemetry (also returned so callers/tests can assert on
/// learning progress without scraping stdout).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainLog {
    /// One-based optimizer step, matching the progress output.
    pub step: usize,
    pub lr: f64,
    pub loss: f64,
    /// Per-channel train accuracy on masked cells.
    pub train_acc: [f64; 4],
    /// Entity placement recall (accuracy on masked non-empty cells) — the honest
    /// "is it learning to build?" signal, immune to the empty-cell majority.
    pub placement_acc: f64,
    /// Mean sampled diffusion time (masking rate) for the batch.
    pub t_mean: f64,
    /// Unweighted negative log-likelihood on masked cells.
    pub nll: f64,
    pub channel_nll: [f64; 4],
    /// Wall-clock seconds since this training run started.
    pub elapsed_seconds: f64,
    pub samples_seen: usize,
    /// Average examples processed per second since the run started.
    pub samples_per_second: f64,
    /// Validation report (only present on validation steps).
    pub val: Option<ReconReport>,
    /// The same validation metrics split by frozen lesson family.
    pub val_by_lesson: BTreeMap<String, ReconReport>,
}

/// Train a denoiser from scratch. Returns the model and the collected logs.
pub fn train<B: AutodiffBackend>(
    cfg: &TrainConfig,
    device: &B::Device,
) -> (Denoiser<B>, Vec<TrainLog>) {
    train_with_observer(cfg, device, |_| {})
}

/// Train while delivering every telemetry record to `observer` immediately.
///
/// This lets the CLI durably append JSONL during long GPU runs; a killed run
/// still retains every completed step instead of losing all metrics at exit.
pub fn train_with_observer<B, F>(
    cfg: &TrainConfig,
    device: &B::Device,
    mut observer: F,
) -> (Denoiser<B>, Vec<TrainLog>)
where
    B: AutodiffBackend,
    F: FnMut(&TrainLog),
{
    let mut model = cfg.model.init::<B>(device);
    let mut optim = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.grad_clip)))
        .init();

    let mut data_rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0xA11CE);
    let mut seed_ctr: u64 = 0;
    let mut logs = Vec::new();
    let validation = (cfg.val_every > 0).then(|| build_validation_set(cfg));
    let started = Instant::now();

    for step in 0..cfg.steps {
        let (grids, observed) =
            train_batch(cfg.grid_size, cfg.batch_size, &mut data_rng, &mut seed_ctr);
        let batch = GridBatch::<B>::from_grids(&grids, Some(&observed), device);

        let (loss_t, stats) = loss(&model, &batch, &cfg.diffusion);

        let lr = lr_at(step, cfg);
        let grads = loss_t.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        let loss_v = loss_t.into_scalar().to_f64();
        let train_acc = [
            stats.channel_acc(0),
            stats.channel_acc(1),
            stats.channel_acc(2),
            stats.channel_acc(3),
        ];

        let is_val = cfg.val_every > 0 && (step + 1) % cfg.val_every == 0;
        let (val, val_by_lesson) = if is_val {
            let (aggregate, by_lesson) = validate::<B>(
                &model,
                cfg,
                validation.as_ref().expect("validation set initialized"),
                device,
            );
            (Some(aggregate), by_lesson)
        } else {
            (None, BTreeMap::new())
        };

        let placement_acc = stats.placement_acc();
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let samples_seen = (step + 1) * cfg.batch_size;
        let samples_per_second = samples_seen as f64 / elapsed_seconds.max(f64::EPSILON);
        if is_val || step == 0 {
            let mut line = format!(
                "step {:>5}/{} | lr {:.2e} | loss {:.4} | place {:.2} | acc[E={:.2} D={:.2} I={:.2} M={:.2}]",
                step + 1,
                cfg.steps,
                lr,
                loss_v,
                placement_acc,
                train_acc[0],
                train_acc[1],
                train_acc[2],
                train_acc[3],
            );
            if let Some(r) = &val {
                line.push_str(&format!(" || VAL {r}"));
            }
            println!("{line}");
            // Flush so progress is visible immediately even when stdout is
            // redirected to a file / pipe (block-buffered otherwise).
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }

        let log = TrainLog {
            step: step + 1,
            lr,
            loss: loss_v,
            train_acc,
            placement_acc,
            t_mean: stats.t_mean,
            nll: stats.nll,
            channel_nll: stats.channel_nll,
            elapsed_seconds,
            samples_seen,
            samples_per_second,
            val,
            val_by_lesson,
        };
        observer(&log);
        logs.push(log);
    }

    (model, logs)
}

/// Warmup-then-cosine learning rate.
fn lr_at(step: usize, cfg: &TrainConfig) -> f64 {
    if step < cfg.warmup {
        return cfg.lr * (step as f64 + 1.0) / cfg.warmup.max(1) as f64;
    }
    let progress =
        (step - cfg.warmup) as f64 / (cfg.steps.saturating_sub(cfg.warmup).max(1)) as f64;
    let cos = 0.5 * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos());
    cfg.lr * cos
}

/// Kinds that can be generated at a given grid size.
fn feasible_kinds(size: usize) -> Vec<LessonKind> {
    LessonKind::all()
        .iter()
        .copied()
        .filter(|k| match k {
            LessonKind::MoveOneItem | LessonKind::MoveOneItemChaos => size >= 3,
            LessonKind::AssemblerLine => size >= 5,
            LessonKind::UndergroundCross => size >= 7,
        })
        .collect()
}

/// Draw a single functional lesson, retrying kinds/seeds until one validates.
fn draw_sample(size: usize, rng: &mut ChaCha8Rng, seed_ctr: &mut u64) -> Sample {
    let kinds = feasible_kinds(size);
    loop {
        let kind = kinds[rng.gen_range(0..kinds.len())];
        let seed = *seed_ctr;
        *seed_ctr += 1;
        if let Some(s) = generate(kind, size, seed) {
            return s;
        }
    }
}

/// A training batch: solution grids + `observed` masks (the protected scaffold is
/// always visible; the diffusion process masks a random subset of the rest).
fn train_batch(
    size: usize,
    batch: usize,
    rng: &mut ChaCha8Rng,
    seed_ctr: &mut u64,
) -> (Vec<Grid>, Vec<Vec<bool>>) {
    let mut grids = Vec::with_capacity(batch);
    let mut observed = Vec::with_capacity(batch);
    for _ in 0..batch {
        let s = draw_sample(size, rng, seed_ctr);
        let obs: Vec<bool> = (0..s.solution.len())
            .map(|i| s.protected.contains(&i))
            .collect();
        grids.push(s.solution);
        observed.push(obs);
    }
    (grids, observed)
}

/// Blank known factories, reconstruct them, and score. This is the
/// always-available "is it really learning?" signal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ValidationSet {
    originals: Vec<Grid>,
    partials: Vec<Grid>,
    observed: Vec<Vec<bool>>,
    kinds: Vec<LessonKind>,
}

/// Build one held-out corpus per run. Its seeds never depend on how much
/// training data has been consumed, so every checkpoint is compared on the
/// exact same tasks.
fn build_validation_set(cfg: &TrainConfig) -> ValidationSet {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x05EE_DF12_EDA7_A5E7);
    let kinds = feasible_kinds(cfg.grid_size);
    let mut seed_ctr = cfg.seed ^ 0x0DD0_0DD0_0DD0_0DD0;
    let mut originals = Vec::with_capacity(cfg.val_batch);
    let mut partials = Vec::with_capacity(cfg.val_batch);
    let mut observed = Vec::with_capacity(cfg.val_batch);
    let mut selected_kinds = Vec::with_capacity(cfg.val_batch);

    for i in 0..cfg.val_batch {
        let kind = kinds[i % kinds.len()];
        let sample = loop {
            let seed = seed_ctr;
            seed_ctr = seed_ctr.wrapping_add(1);
            if let Some(sample) = generate(kind, cfg.grid_size, seed) {
                break sample;
            }
        };
        let (partial, obs) = sample.blank(None, &mut rng);
        originals.push(sample.solution);
        partials.push(partial);
        observed.push(obs);
        selected_kinds.push(kind);
    }

    ValidationSet {
        originals,
        partials,
        observed,
        kinds: selected_kinds,
    }
}

fn validate<B: AutodiffBackend>(
    model: &Denoiser<B>,
    cfg: &TrainConfig,
    validation: &ValidationSet,
    device: &B::Device,
) -> (ReconReport, BTreeMap<String, ReconReport>) {
    // Use the inner (non-autodiff) backend for inference.
    let inner = model.valid();
    let sample_cfg = SampleConfig {
        steps: cfg.sample_steps,
        temperature: 0.0,
        seed: 0,
    };
    let recon = reconstruct(
        &inner,
        &validation.partials,
        &validation.observed,
        &sample_cfg,
        device,
    );
    let aggregate = reconstruction_report(&validation.originals, &recon, &validation.observed);
    let mut by_lesson = BTreeMap::new();
    for &kind in feasible_kinds(cfg.grid_size).iter() {
        let indexes: Vec<usize> = validation
            .kinds
            .iter()
            .enumerate()
            .filter_map(|(i, candidate)| (*candidate == kind).then_some(i))
            .collect();
        let originals: Vec<Grid> = indexes
            .iter()
            .map(|&i| validation.originals[i].clone())
            .collect();
        let reconstructed: Vec<Grid> = indexes.iter().map(|&i| recon[i].clone()).collect();
        let observed: Vec<Vec<bool>> = indexes
            .iter()
            .map(|&i| validation.observed[i].clone())
            .collect();
        by_lesson.insert(
            kind.name().to_owned(),
            reconstruction_report(&originals, &reconstructed, &observed),
        );
    }
    (aggregate, by_lesson)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuAutodiff;

    #[test]
    fn tiny_training_run_reduces_loss() {
        type B = CpuAutodiff;
        let device = Default::default();
        let cfg = TrainConfig {
            grid_size: 11,
            steps: 40,
            batch_size: 8,
            warmup: 5,
            val_every: 0, // skip validation in the unit test (fast)
            model: DenoiserConfig::new().with_hidden(24).with_blocks(3),
            ..Default::default()
        };
        let (_model, logs) = train::<B>(&cfg, &device);
        let first = logs.first().unwrap().loss;
        let last = logs.last().unwrap().loss;
        assert!(last.is_finite());
        // A few dozen steps should already move the loss down noticeably.
        assert!(last < first, "loss did not decrease: {first} -> {last}");
    }

    #[test]
    fn validation_corpus_is_frozen_for_a_run() {
        let cfg = TrainConfig {
            val_batch: 12,
            seed: 42,
            ..Default::default()
        };
        assert_eq!(build_validation_set(&cfg), build_validation_set(&cfg));

        let other = TrainConfig {
            seed: 43,
            ..cfg.clone()
        };
        assert_ne!(build_validation_set(&other), build_validation_set(&cfg));
    }
}
