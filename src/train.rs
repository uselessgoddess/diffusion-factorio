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
use crate::factory_gen::{
    generate, Canvas, LessonKind, Sample, DEFAULT_CANVAS_MAX, DEFAULT_CANVAS_MIN,
};
use crate::metrics::{reconstruction_report, ReconReport};
use crate::model::{Denoiser, DenoiserConfig};
use crate::sample::{reconstruct, SampleConfig};
use crate::world::Grid;

/// Everything needed to run a training session.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// The canvas shapes to train on. One is drawn per batch, uniformly.
    ///
    /// This used to be a single `grid_size: usize` and every lesson was built on
    /// a square of that side, which is the whole of cause 5 in
    /// `docs/GENERALIZATION.md`: the issue trains on 11×11, infers on 13×9, and
    /// the same task drops from 3.118 items/s to 0.478 for no reason but the
    /// shape of the canvas around it. A model shown one shape has no way to know
    /// the shape was not part of the task.
    ///
    /// Drawn per *batch* rather than per sample because `GridBatch` is a single
    /// tensor and asserts against a ragged batch. That costs nothing: the shape
    /// still varies across a run, which is the axis the model has to generalize
    /// over.
    pub canvases: Vec<Canvas>,
    /// Number of optimizer steps.
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Linear warmup steps before cosine decay.
    pub warmup: usize,
    pub grad_clip: f32,
    /// Run validation (and log) every this many steps.
    pub val_every: usize,
    /// Held-out factories per validation pass.
    ///
    /// This is the resolution of every headline number, and small values are
    /// dishonest: for an all-successes run the 95% lower bound on the true rate
    /// is `0.05^(1/n)`, so 64/64 perfect only proves >95.4%, and per-lesson
    /// (n/4) 16/16 only proves >82.9%. Validation costs ~1.5% of a run's wall
    /// clock, so buying resolution here is nearly free.
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
            canvases: Canvas::pool(DEFAULT_CANVAS_MIN, DEFAULT_CANVAS_MAX),
            steps: 2000,
            batch_size: 32,
            lr: 3e-4,
            warmup: 100,
            grad_clip: 1.0,
            val_every: 100,
            val_batch: 512,
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
    /// Validation on the same factories with only the source/sink anchors left
    /// visible, so the model must build rather than inpaint. `functional` is the
    /// metric to read here: many layouts are valid, so `exact` understates it.
    pub val_scratch: Option<ReconReport>,
    /// The from-scratch metrics split by frozen lesson family.
    pub val_scratch_by_lesson: BTreeMap<String, ReconReport>,
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

    assert!(
        !cfg.canvases.is_empty(),
        "a curriculum with no canvas to build on"
    );
    for step in 0..cfg.steps {
        // One shape for the whole batch: `GridBatch` is one tensor and rejects a
        // ragged one. Across steps the shape still varies, which is the axis
        // that matters.
        let canvas = cfg.canvases[data_rng.gen_range(0..cfg.canvases.len())];
        let (grids, observed) = train_batch(canvas, cfg.batch_size, &mut data_rng, &mut seed_ctr);
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
        let report = is_val.then(|| {
            validate::<B>(
                &model,
                cfg,
                validation.as_ref().expect("validation set initialized"),
                device,
            )
        });
        let (val, val_by_lesson, val_scratch, val_scratch_by_lesson) = match report {
            Some(r) => (
                Some(r.inpaint),
                r.inpaint_by_lesson,
                Some(r.scratch),
                r.scratch_by_lesson,
            ),
            None => (None, BTreeMap::new(), None, BTreeMap::new()),
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
            // The from-scratch score is the one that says whether the model can
            // build a factory rather than fill gaps in a given one, so it is
            // worth the width in the log.
            if let Some(r) = &val_scratch {
                line.push_str(&format!(" || SCRATCH {r}"));
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
            val_scratch,
            val_scratch_by_lesson,
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

/// Kinds that can be generated on a given canvas.
///
/// Asked per axis, not against the longer side. That is the difference between
/// a 13×9 canvas offering the whole curriculum and offering six of eight
/// families: a circuit line is 11×5 and a shared line 11×7, and both fit 13×9
/// with room to spare — under one square number they were billed as needing 11
/// rows apiece and ruled out.
pub fn feasible_kinds(canvas: Canvas) -> Vec<LessonKind> {
    LessonKind::all()
        .iter()
        .copied()
        .filter(|k| k.fits(canvas))
        .collect()
}

/// Every kind some canvas in the pool can hold.
///
/// A union rather than an intersection: a pool is a set of shapes the model must
/// handle, and a family that only fits the wider ones is still part of the
/// curriculum. `draw_sample` never offers a kind to a canvas it does not fit.
pub fn curriculum_kinds(canvases: &[Canvas]) -> Vec<LessonKind> {
    LessonKind::all()
        .iter()
        .copied()
        .filter(|k| canvases.iter().any(|c| k.fits(*c)))
        .collect()
}

/// Draw a single functional lesson, retrying kinds/seeds until one validates.
fn draw_sample(canvas: Canvas, rng: &mut ChaCha8Rng, seed_ctr: &mut u64) -> Sample {
    let kinds = feasible_kinds(canvas);
    loop {
        let kind = kinds[rng.gen_range(0..kinds.len())];
        let seed = *seed_ctr;
        *seed_ctr += 1;
        if let Some(s) = generate(kind, canvas, seed) {
            return s;
        }
    }
}

/// A training batch: solution grids + `observed` masks (the protected scaffold is
/// always visible; the diffusion process masks a random subset of the rest).
fn train_batch(
    canvas: Canvas,
    batch: usize,
    rng: &mut ChaCha8Rng,
    seed_ctr: &mut u64,
) -> (Vec<Grid>, Vec<Vec<bool>>) {
    let mut grids = Vec::with_capacity(batch);
    let mut observed = Vec::with_capacity(batch);
    for _ in 0..batch {
        let s = draw_sample(canvas, rng, seed_ctr);
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
    /// The same factories with everything but the source/sink anchors blanked,
    /// so the model has to build them rather than fill in a few gaps. See
    /// [`Sample::blank_to_scaffold`].
    scratch_partials: Vec<Grid>,
    scratch_observed: Vec<Vec<bool>>,
    kinds: Vec<LessonKind>,
}

/// Build one held-out corpus per run. Its seeds never depend on how much
/// training data has been consumed, so every checkpoint is compared on the
/// exact same tasks.
fn build_validation_set(cfg: &TrainConfig) -> ValidationSet {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x05EE_DF12_EDA7_A5E7);
    let kinds = curriculum_kinds(&cfg.canvases);
    let mut seed_ctr = cfg.seed ^ 0x0DD0_0DD0_0DD0_0DD0;
    let mut originals = Vec::with_capacity(cfg.val_batch);
    let mut partials = Vec::with_capacity(cfg.val_batch);
    let mut observed = Vec::with_capacity(cfg.val_batch);
    let mut scratch_partials = Vec::with_capacity(cfg.val_batch);
    let mut scratch_observed = Vec::with_capacity(cfg.val_batch);
    let mut selected_kinds = Vec::with_capacity(cfg.val_batch);

    for i in 0..cfg.val_batch {
        let kind = kinds[i % kinds.len()];
        // Cycle the shapes this kind fits on, independently per kind, so the
        // held-out corpus scores every family on every canvas it can occupy
        // rather than on one lucky shape. A validation set that is square when
        // the curriculum is not would report a generalization the run never has
        // to earn.
        let shapes: Vec<Canvas> = cfg
            .canvases
            .iter()
            .copied()
            .filter(|c| kind.fits(*c))
            .collect();
        let canvas = shapes[(i / kinds.len()) % shapes.len()];
        let sample = loop {
            let seed = seed_ctr;
            seed_ctr = seed_ctr.wrapping_add(1);
            if let Some(sample) = generate(kind, canvas, seed) {
                break sample;
            }
        };
        let (partial, obs) = sample.blank(None, &mut rng);
        let (scratch_partial, scratch_obs) = sample.blank_to_scaffold();
        originals.push(sample.solution);
        partials.push(partial);
        observed.push(obs);
        scratch_partials.push(scratch_partial);
        scratch_observed.push(scratch_obs);
        selected_kinds.push(kind);
    }

    ValidationSet {
        originals,
        partials,
        observed,
        scratch_partials,
        scratch_observed,
        kinds: selected_kinds,
    }
}

/// One validation pass in both modes.
struct Validation {
    /// Fill the gaps in a given scaffold (the historical metric).
    inpaint: ReconReport,
    inpaint_by_lesson: BTreeMap<String, ReconReport>,
    /// Build the factory given only the source/sink anchors.
    scratch: ReconReport,
    scratch_by_lesson: BTreeMap<String, ReconReport>,
}

fn validate<B: AutodiffBackend>(
    model: &Denoiser<B>,
    cfg: &TrainConfig,
    validation: &ValidationSet,
    device: &B::Device,
) -> Validation {
    // Use the inner (non-autodiff) backend for inference.
    let inner = model.valid();
    let sample_cfg = SampleConfig {
        steps: cfg.sample_steps,
        temperature: 0.0,
        seed: 0,
    };

    let run = |partials: &[Grid], observed: &[Vec<bool>]| {
        // The corpus is no longer one shape, and `GridBatch` is one tensor: it
        // asserts against a ragged batch. So reconstruct shape group by shape
        // group and reassemble in the original order — every report below reads
        // the same indices it always did.
        let mut shapes: Vec<(usize, usize)> =
            partials.iter().map(|g| (g.width, g.height)).collect();
        shapes.sort_unstable();
        shapes.dedup();
        let mut recon: Vec<Option<Grid>> = vec![None; partials.len()];
        for shape in shapes {
            let idx: Vec<usize> = (0..partials.len())
                .filter(|&i| (partials[i].width, partials[i].height) == shape)
                .collect();
            let group: Vec<Grid> = idx.iter().map(|&i| partials[i].clone()).collect();
            let group_obs: Vec<Vec<bool>> = idx.iter().map(|&i| observed[i].clone()).collect();
            let out = reconstruct(&inner, &group, &group_obs, &sample_cfg, device);
            for (&i, grid) in idx.iter().zip(out) {
                recon[i] = Some(grid);
            }
        }
        let recon: Vec<Grid> = recon
            .into_iter()
            .map(|g| g.expect("every validation grid belongs to a shape group"))
            .collect();

        let aggregate = reconstruction_report(&validation.originals, &recon, observed);
        let mut by_lesson = BTreeMap::new();
        for &kind in curriculum_kinds(&cfg.canvases).iter() {
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
            let obs: Vec<Vec<bool>> = indexes.iter().map(|&i| observed[i].clone()).collect();
            by_lesson.insert(
                kind.name().to_owned(),
                reconstruction_report(&originals, &reconstructed, &obs),
            );
        }
        (aggregate, by_lesson)
    };

    let (inpaint, inpaint_by_lesson) = run(&validation.partials, &validation.observed);
    let (scratch, scratch_by_lesson) =
        run(&validation.scratch_partials, &validation.scratch_observed);
    Validation {
        inpaint,
        inpaint_by_lesson,
        scratch,
        scratch_by_lesson,
    }
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
            canvases: vec![Canvas::square(11)],
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

    /// The default curriculum has to teach the whole vocabulary *on the shape the
    /// issue infers on*, not merely somewhere in the pool. A canvas that only the
    /// simple families fit is a canvas where the model has never been asked to
    /// compose anything, and 13×9 is exactly the case the issue reports.
    #[test]
    fn the_default_curriculum_teaches_every_family_on_the_inference_canvas() {
        let cfg = TrainConfig::default();
        assert!(
            cfg.canvases.contains(&Canvas::new(13, 9)),
            "the default pool never draws the shape the issue infers on"
        );
        assert_eq!(
            feasible_kinds(Canvas::new(13, 9)).len(),
            LessonKind::all().len(),
            "13x9 is missing lessons the model is expected to know there"
        );
        assert_eq!(
            curriculum_kinds(&cfg.canvases).len(),
            LessonKind::all().len()
        );
    }

    /// Every canvas in the default pool must be able to hold *something*, or
    /// `draw_sample` indexes an empty `kinds` and the run panics on whichever
    /// step happens to draw that shape — a failure that would surface at minute
    /// forty of a GPU run rather than here.
    #[test]
    fn every_canvas_in_the_default_pool_can_hold_a_lesson() {
        for canvas in TrainConfig::default().canvases {
            assert!(
                !feasible_kinds(canvas).is_empty(),
                "no lesson fits {canvas}, which the curriculum still draws"
            );
        }
    }

    /// A batch is one tensor and `GridBatch::from_grids` asserts against a ragged
    /// one, so the per-batch draw is load-bearing rather than an optimization.
    #[test]
    fn a_batch_is_drawn_at_one_shape() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut seed_ctr = 0;
        let canvas = Canvas::new(13, 9);
        let (grids, _) = train_batch(canvas, 8, &mut rng, &mut seed_ctr);
        assert!(grids
            .iter()
            .all(|g| (g.width, g.height) == (canvas.width, canvas.height)));
    }
}
