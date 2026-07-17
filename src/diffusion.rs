//! Masked (absorbing-state) discrete diffusion.
//!
//! We use the absorbing/MASK formulation (the family behind MaskGIT, D3PM-
//! absorbing, MDLM and DiffusionGemma-style text diffusion) because it maps
//! cleanly onto the factorion task: conditioning on a partial factory is just
//! *inpainting* — the observed cells are never masked, the missing entities are
//! MASK tokens the model fills in. See `docs/DESIGN.md` for the full rationale.
//!
//! Forward process (continuous time `t in (0,1]`, linear schedule): each
//! generative cell is independently replaced by MASK with probability `t`.
//! At `t -> 1` the whole (non-observed) grid is masked; at `t -> 0` nothing is.
//!
//! Reverse process: the denoiser predicts `x_0` from the masked grid; we
//! ancestrally unmask cells as `t` decreases (see [`sample`](crate::sample)).

use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::{Distribution, Int};

use crate::data::GridBatch;
use crate::model::Denoiser;
use crate::world::{Entity, N_CHANNELS, VOCAB};

#[derive(Config, Debug)]
pub struct DiffusionConfig {
    /// Use the MDLM continuous-time ELBO weight `1/t` (principled likelihood
    /// bound) instead of a plain mean over masked cells (MaskGIT objective,
    /// lower variance / more robust for short smoke runs).
    #[config(default = false)]
    pub elbo_weight: bool,
    /// Clamp `t` into `[t_min, 1]` to bound the `1/t` weight's variance.
    #[config(default = 0.02)]
    pub t_min: f64,
    /// Fraction of examples trained at exactly `t = 1`, matching the fully
    /// masked state from which reverse diffusion starts.
    ///
    /// A continuous draw from `[t_min, 1)` has zero probability of producing
    /// that state. On a board with many answer cells, even a draw near one is
    /// overwhelmingly unlikely to mask every cell at once, so scratch sampling
    /// was an out-of-distribution input despite being the main evaluation mode.
    #[config(default = 0.25)]
    pub scratch_probability: f64,
    /// Loss weight multiplier for cells whose target entity is *not* `Empty`.
    ///
    /// The entity channel is ~95% `Empty`, so an unweighted objective is trivially
    /// minimized by predicting empty everywhere (the model never learns to place
    /// structure). Up-weighting the rare non-empty cells fixes this dominant
    /// bottleneck — see `docs/ROADMAP.md`. `1.0` disables the reweighting.
    #[config(default = 8.0)]
    pub structure_weight: f64,
    /// Experimental loss multiplier for an assembler anchor, after the general
    /// non-empty-cell weight. `1.0` is deliberately neutral: weighting cannot
    /// repair a machine target chosen independently of the visible task, and
    /// amplifying that label noise harms the routing lessons.
    #[config(default = 1.0)]
    pub assembler_weight: f64,
}

/// Detached statistics from a training step, for metrics/logging.
#[derive(Clone, Debug, Default)]
pub struct StepStats {
    /// Correctly-denoised cells per channel (argmax == target, on masked cells).
    pub correct: [f64; N_CHANNELS],
    /// Number of masked cells (denominator for accuracy).
    pub masked: f64,
    /// Correctly-denoised *entity* on masked cells whose target is non-empty.
    /// This is the honest "is it learning to place structure?" signal —
    /// immune to the empty-cell majority that inflates `channel_acc(0)`.
    pub placement_correct: f64,
    /// Masked cells whose target entity is non-empty (denominator above).
    pub placement_total: f64,
    /// Correct entity predictions specifically on assembler anchors.
    pub assembler_correct: f64,
    /// Masked assembler anchors (denominator for assembler recall and recipe
    /// accuracy).
    pub assembler_total: f64,
    /// Correct item/recipe predictions on masked assembler anchors.
    pub recipe_correct: f64,
    /// Mean masking rate `t` this step.
    pub t_mean: f64,
    /// Mean per-cell NLL over masked cells (unweighted).
    pub nll: f64,
    /// Mean unweighted NLL for each categorical channel.
    pub channel_nll: [f64; N_CHANNELS],
}

impl StepStats {
    pub fn channel_acc(&self, c: usize) -> f64 {
        if self.masked > 0.0 {
            self.correct[c] / self.masked
        } else {
            0.0
        }
    }
    /// Entity accuracy restricted to non-empty target cells (placement recall).
    pub fn placement_acc(&self) -> f64 {
        if self.placement_total > 0.0 {
            self.placement_correct / self.placement_total
        } else {
            0.0
        }
    }
    pub fn assembler_acc(&self) -> f64 {
        if self.assembler_total > 0.0 {
            self.assembler_correct / self.assembler_total
        } else {
            0.0
        }
    }
    pub fn recipe_acc(&self) -> f64 {
        if self.assembler_total > 0.0 {
            self.recipe_correct / self.assembler_total
        } else {
            0.0
        }
    }
}

/// The masking applied to a batch: the model input and which cells were masked.
pub struct Masked<B: Backend> {
    /// Model-input tokens with masked cells set to each channel's MASK id.
    pub input: Tensor<B, 4, Int>,
    /// `[batch, H, W]` float in {0,1}: 1 where a cell was masked.
    pub mask: Tensor<B, 3>,
    /// `[batch]` sampled masking rate.
    pub t: Tensor<B, 1>,
}

fn loss_weights<B: Backend>(
    entity_target: Tensor<B, 3, Int>,
    mask: Tensor<B, 3>,
    cfg: &DiffusionConfig,
) -> Tensor<B, 3> {
    let non_empty = entity_target.clone().greater_elem(0).float();
    let assembler = entity_target.equal_elem(Entity::Assembler as i32).float();
    let structure = non_empty
        .mul_scalar(cfg.structure_weight - 1.0)
        .add_scalar(1.0);
    let machine = assembler
        .mul_scalar(cfg.assembler_weight - 1.0)
        .add_scalar(1.0);
    structure.mul(machine).mul(mask)
}

/// Apply the forward (noising) process to a clean batch.
///
/// Observed cells (`batch.observed`) are never masked — that is what makes the
/// process conditional / an inpainter.
pub fn apply_masking<B: Backend>(batch: &GridBatch<B>, cfg: &DiffusionConfig) -> Masked<B> {
    let device = batch.tokens.device();
    let [n, _c, h, w] = batch.tokens.dims();

    assert!(
        (0.0..=1.0).contains(&cfg.scratch_probability),
        "scratch_probability must be in [0, 1]"
    );

    // Most examples use t ~ U(t_min, 1). A configurable fraction are set to
    // exactly one so training includes the reverse process's initial state.
    let random_t = Tensor::<B, 1>::random([n], Distribution::Uniform(cfg.t_min, 1.0), &device);
    let scratch = Tensor::<B, 1>::random([n], Distribution::Uniform(0.0, 1.0), &device)
        .lower_elem(cfg.scratch_probability)
        .float();
    let t = random_t.mul(scratch.clone().neg().add_scalar(1.0)) + scratch;
    let t_full = Tensor::<B, 3>::zeros([n, h, w], &device) + t.clone().reshape([n, 1, 1]);

    // Bernoulli(t) per cell, then exclude observed cells.
    let u = Tensor::<B, 3>::random([n, h, w], Distribution::Uniform(0.0, 1.0), &device);
    let not_observed = batch.observed.clone().float().neg().add_scalar(1.0); // 1 - observed
    let mask = u.lower(t_full).float().mul(not_observed); // [n,h,w] in {0,1}
    let mask_i = mask.clone().int();
    let keep_i = mask_i.clone().neg().add_scalar(1); // 1 - mask

    // Replace masked cells with the per-channel MASK id.
    let mut chans: Vec<Tensor<B, 4, Int>> = Vec::with_capacity(N_CHANNELS);
    for (c, &vocab) in VOCAB.iter().enumerate() {
        let clean = batch
            .tokens
            .clone()
            .slice([0..n, c..c + 1, 0..h, 0..w])
            .reshape([n, h, w]);
        let mask_id = vocab as i32;
        let noised = clean.mul(keep_i.clone()) + mask_i.clone().mul_scalar(mask_id);
        chans.push(noised.reshape([n, 1, h, w]));
    }
    let input = Tensor::cat(chans, 1);

    Masked { input, mask, t }
}

/// Compute the masked-denoising loss for a batch and return `(loss, stats)`.
///
/// Loss = for every masked cell, cross-entropy of the predicted `x_0`
/// distribution against the true category, summed over the coupled channels.
/// With `elbo_weight` each sample is weighted by `1/t` (MDLM NELBO estimator);
/// otherwise it is a mean over masked cells.
pub fn loss<B: Backend>(
    model: &Denoiser<B>,
    batch: &GridBatch<B>,
    cfg: &DiffusionConfig,
) -> (Tensor<B, 1>, StepStats) {
    let masked = apply_masking(batch, cfg);
    let logits = model.forward(masked.input, batch.obstacle.clone(), masked.t.clone());

    let [n, _c, h, w] = batch.tokens.dims();
    let device = batch.tokens.device();
    let mask = masked.mask; // [n,h,w]

    // Per-cell loss weight: up-weight masked cells whose target entity is
    // non-empty, to counter the empty-cell majority (see `structure_weight`).
    let entity_target = batch
        .tokens
        .clone()
        .slice([0..n, 0..1, 0..h, 0..w])
        .reshape([n, h, w]);
    let non_empty = entity_target.clone().greater_elem(0).float(); // [n,h,w] in {0,1}
    let assembler = entity_target
        .clone()
        .equal_elem(Entity::Assembler as i32)
        .float();
    let assembler_mask = assembler.mul(mask.clone());
    let weight = loss_weights(entity_target, mask.clone(), cfg); // 0 on unmasked cells
    let w_sum = weight.clone().sum(); // total weight (scalar tensor)
    let n_masked = mask.clone().sum();

    // Per-sample weighted NLL over masked cells and channels.
    let mut nll_per_sample = Tensor::<B, 1>::zeros([n], &device);
    let mut stats = StepStats::default();
    for (c, logit) in logits.iter().enumerate() {
        let target = batch
            .tokens
            .clone()
            .slice([0..n, c..c + 1, 0..h, 0..w])
            .reshape([n, h, w]);
        let logp = log_softmax(logit.clone(), 1); // [n, K, h, w]
                                                  // gather log p(true class)
        let idx = target.clone().reshape([n, 1, h, w]);
        let chosen = logp.gather(1, idx).reshape([n, h, w]); // log p(true)
        let raw_nll = chosen.neg(); // [n,h,w] per-cell -log p(true)
        let nll = raw_nll.clone().mul(weight.clone()); // weighted, 0 on unmasked
        nll_per_sample = nll_per_sample + nll.sum_dim(2).sum_dim(1).reshape([n]);

        // Accuracy (detached, unweighted counts over masked cells).
        let pred = logit.clone().argmax(1).reshape([n, h, w]); // [n,h,w] int
        let hit = pred.equal(target).float();
        stats.correct[c] = scalar(hit.clone().mul(mask.clone()).sum());
        if c == 0 {
            // Placement recall: entity hits on masked, non-empty target cells.
            let placement_mask = non_empty.clone().mul(mask.clone());
            stats.placement_correct = scalar(hit.clone().mul(placement_mask.clone()).sum());
            stats.placement_total = scalar(placement_mask.sum());
            stats.assembler_correct = scalar(hit.mul(assembler_mask.clone()).sum());
            stats.assembler_total = scalar(assembler_mask.clone().sum());
        } else if c == 2 {
            // The item channel on an assembler is its recipe. Accuracy over all
            // item cells is mostly `None` and concealed the recipe blind spot.
            stats.recipe_correct = scalar(hit.mul(assembler_mask.clone()).sum());
        }
        let channel_nll = scalar(raw_nll.mul(mask.clone()).sum());
        stats.channel_nll[c] = channel_nll;
        stats.nll += channel_nll;
    }
    stats.masked = scalar(n_masked.clone());
    stats.t_mean = scalar(masked.t.clone().mean());
    if stats.masked > 0.0 {
        stats.nll /= stats.masked;
        for value in &mut stats.channel_nll {
            *value /= stats.masked;
        }
    }

    let d = (h * w) as f64;
    let loss = if cfg.elbo_weight {
        // (1/t) * (sum_masked weighted nll) / D, averaged over batch.
        let w_ = masked.t.recip(); // [n]
        nll_per_sample.mul(w_).div_scalar(d).mean()
    } else {
        // Weighted mean cross-entropy per masked cell (robust default).
        let denom = w_sum.clamp_min(1.0);
        nll_per_sample.sum().div(denom).reshape([1])
    };

    (loss, stats)
}

fn scalar<B: Backend, const D: usize>(t: Tensor<B, D>) -> f64 {
    t.into_scalar().to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::factory_gen::{generate, Canvas, LessonKind};
    use crate::model::DenoiserConfig;

    #[test]
    fn assembler_anchors_get_dedicated_loss_weight() {
        type B = CpuBackend;
        let device = Default::default();
        let entity = Tensor::<B, 1, Int>::from_ints(
            [
                Entity::Empty as i32,
                Entity::TransportBelt as i32,
                Entity::Assembler as i32,
            ],
            &device,
        )
        .reshape([1, 1, 3]);
        let mask = Tensor::<B, 3>::ones([1, 1, 3], &device);
        let cfg = DiffusionConfig::new();

        let values: Vec<f32> = loss_weights(entity, mask, &cfg)
            .to_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();
        assert_eq!(values, vec![1.0, 8.0, 8.0]);
    }

    #[test]
    fn masking_leaves_observed_cells_untouched() {
        type B = CpuBackend;
        let device = Default::default();
        let s = generate(LessonKind::MoveOneItem, Canvas::square(11), 5).unwrap();
        let observed: Vec<bool> = (0..s.solution.len())
            .map(|i| s.protected.contains(&i))
            .collect();
        let batch = GridBatch::<B>::from_grids(
            std::slice::from_ref(&s.solution),
            Some(std::slice::from_ref(&observed)),
            &device,
        );
        let cfg = DiffusionConfig::new()
            .with_t_min(0.9)
            .with_scratch_probability(0.0);
        let masked = apply_masking(&batch, &cfg); // high t -> most non-observed masked
        let mask_data: Vec<f32> = masked.mask.to_data().convert::<f32>().into_vec().unwrap();
        // Observed cells are never masked.
        for (i, &o) in observed.iter().enumerate() {
            if o {
                assert_eq!(mask_data[i], 0.0);
            }
        }
    }

    #[test]
    fn scratch_examples_mask_every_answer_but_never_the_task_anchors() {
        type B = CpuBackend;
        let device = Default::default();
        let s = generate(LessonKind::AssemblerLine, Canvas::square(11), 5).unwrap();
        let (_, observed) = s.blank_to_scaffold();
        let batch = GridBatch::<B>::from_grids(
            std::slice::from_ref(&s.solution),
            Some(std::slice::from_ref(&observed)),
            &device,
        );
        let cfg = DiffusionConfig::new().with_scratch_probability(1.0);
        let masked = apply_masking(&batch, &cfg);
        let mask: Vec<f32> = masked.mask.to_data().convert::<f32>().into_vec().unwrap();
        let times: Vec<f32> = masked.t.to_data().convert::<f32>().into_vec().unwrap();

        assert_eq!(times, vec![1.0]);
        for (i, &anchor) in observed.iter().enumerate() {
            assert_eq!(
                mask[i],
                if anchor { 0.0 } else { 1.0 },
                "wrong scratch mask at cell {i}"
            );
        }
    }

    #[test]
    fn loss_is_finite_and_positive() {
        type B = CpuBackend;
        let device = Default::default();
        let s = generate(LessonKind::MoveOneItem, Canvas::square(11), 1).unwrap();
        let batch = GridBatch::<B>::from_grids(std::slice::from_ref(&s.solution), None, &device);
        let model = DenoiserConfig::new()
            .with_hidden(16)
            .with_blocks(2)
            .init::<B>(&device);
        let (l, stats) = loss(&model, &batch, &DiffusionConfig::new());
        let v = l.into_scalar().to_f64();
        assert!(v.is_finite() && v > 0.0, "loss={v}");
        assert!(stats.masked > 0.0);
    }
}
