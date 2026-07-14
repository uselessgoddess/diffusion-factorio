//! Reverse (denoising) sampler — the inference side of the diffusion model.
//!
//! We use confidence-based iterative decoding (MaskGIT-style): start from the
//! partial factory (observed cells fixed, everything else MASK), and over `steps`
//! rounds ask the denoiser to predict `x_0` everywhere, then *commit* the most
//! confident still-masked cells and re-predict the rest. Observed cells are held
//! fixed the whole way, so this is exactly conditional inpainting.
//!
//! The whole loop runs host-side (small validation batches): it keeps a plain
//! `Vec<i32>` of tokens, rebuilds the input tensor each round and reads softmax
//! probabilities back. That keeps the reveal/confidence logic transparent — the
//! user asked for inference we can always inspect and validate.

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{Int, TensorData};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::data::grid_from_ids;
use crate::model::Denoiser;
use crate::world::{Grid, N_CHANNELS, VOCAB};

/// Sampler settings.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    /// Number of unmasking rounds. More rounds = finer-grained commitment.
    pub steps: usize,
    /// Softmax temperature for the per-cell category draw. `0.0` = greedy argmax
    /// (deterministic; used for reconstruction validation).
    pub temperature: f64,
    /// RNG seed (only used when `temperature > 0`).
    pub seed: u64,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            steps: 12,
            temperature: 0.0,
            seed: 0,
        }
    }
}

/// A reconstruction plus spatial uncertainty captured when each generated cell
/// was committed by the reverse-diffusion schedule.
#[derive(Clone, Debug)]
pub struct ReconstructionDiagnostics {
    pub grid: Grid,
    /// Mean probability assigned to the selected category across four heads.
    pub confidence: Vec<f32>,
    /// Mean normalized categorical entropy across four heads (`0..=1`).
    pub entropy: Vec<f32>,
    /// One-based reveal round; zero denotes an observed conditioning cell.
    pub reveal_step: Vec<usize>,
}

/// A partial factory to be completed: which cells are given, and the ground-truth
/// (only read at observed cells) plus obstacle conditioning.
struct HostInputs {
    n: usize,
    h: usize,
    w: usize,
    /// `[n, N_CHANNELS, H, W]` category ids (ground truth; read only at observed).
    tokens: Vec<i32>,
    /// `[n, H, W]` obstacle flags.
    obstacle: Vec<f32>,
    /// `[n, H, W]` observed flags.
    observed: Vec<bool>,
}

/// Iteratively denoise `partials` into completed grids.
///
/// * `partials`: for each sample, the (already blanked) grid — only its
///   *observed* cells are read; masked cells are ignored and regenerated.
/// * `observed`: per-sample `true`=given / `false`=to-generate, row-major.
///
/// Returns one completed [`Grid`] per input.
pub fn reconstruct<B: Backend>(
    model: &Denoiser<B>,
    partials: &[Grid],
    observed: &[Vec<bool>],
    cfg: &SampleConfig,
    device: &B::Device,
) -> Vec<Grid> {
    reconstruct_with_diagnostics(model, partials, observed, cfg, device)
        .into_iter()
        .map(|result| result.grid)
        .collect()
}

/// Reconstruct factories and retain confidence, entropy and reveal-time maps.
pub fn reconstruct_with_diagnostics<B: Backend>(
    model: &Denoiser<B>,
    partials: &[Grid],
    observed: &[Vec<bool>],
    cfg: &SampleConfig,
    device: &B::Device,
) -> Vec<ReconstructionDiagnostics> {
    assert_eq!(partials.len(), observed.len());
    if partials.is_empty() {
        return Vec::new();
    }
    let inputs = to_host(partials, observed);
    let HostInputs {
        n,
        h,
        w,
        tokens,
        obstacle,
        observed,
    } = inputs;
    let plane = h * w;

    // Working state: observed cells keep their id; masked cells start at MASK.
    let mut cur = tokens.clone();
    let mut masked = vec![false; n * plane]; // per (sample, cell)
    let mut confidence = vec![1.0f32; n * plane];
    let mut entropy = vec![0.0f32; n * plane];
    let mut reveal_step = vec![0usize; n * plane];
    for s in 0..n {
        for cell in 0..plane {
            let obs = observed[s * plane + cell];
            masked[s * plane + cell] = !obs;
            if !obs {
                for c in 0..N_CHANNELS {
                    cur[(s * N_CHANNELS + c) * plane + cell] = VOCAB[c] as i32; // MASK id
                }
            }
        }
    }
    let masked0: Vec<usize> = (0..n)
        .map(|s| (0..plane).filter(|&cell| masked[s * plane + cell]).count())
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let steps = cfg.steps.max(1);

    for step in 0..steps {
        // Current diffusion time (near 1 at the start, 0 at the end).
        let t_val = (std::f64::consts::FRAC_PI_2 * step as f64 / steps as f64).cos();
        let probs = predict(model, &cur, &obstacle, [n, h, w], t_val, device);

        for s in 0..n {
            // How many still-masked cells to reveal this round (cosine schedule).
            let remaining = (0..plane).filter(|&cell| masked[s * plane + cell]).count();
            if remaining == 0 {
                continue;
            }
            let frac = (std::f64::consts::FRAC_PI_2 * (step + 1) as f64 / steps as f64).cos();
            let should_remain = if step + 1 == steps {
                0
            } else {
                (masked0[s] as f64 * frac).round() as usize
            };
            let reveal = remaining
                .saturating_sub(should_remain)
                .max(1)
                .min(remaining);

            // Score every still-masked cell; pick the `reveal` most confident.
            let mut scored: Vec<(CellPrediction, usize)> = Vec::with_capacity(remaining);
            for cell in 0..plane {
                if !masked[s * plane + cell] {
                    continue;
                }
                let prediction = decode_cell(&probs, s, cell, plane, cfg, &mut rng);
                scored.push((prediction, cell));
            }
            // Highest score first.
            scored.sort_by(|a, b| {
                b.0.score
                    .partial_cmp(&a.0.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (prediction, cell) in scored.into_iter().take(reveal) {
                for c in 0..N_CHANNELS {
                    cur[(s * N_CHANNELS + c) * plane + cell] = prediction.ids[c] as i32;
                }
                confidence[s * plane + cell] = prediction.confidence;
                entropy[s * plane + cell] = prediction.entropy;
                reveal_step[s * plane + cell] = step + 1;
                masked[s * plane + cell] = false;
            }
        }
    }

    // Decode back into grids.
    (0..n)
        .map(|s| {
            let base = s * N_CHANNELS * plane;
            let ids = &cur[base..base + N_CHANNELS * plane];
            let obs_flags: Vec<bool> = (0..plane)
                .map(|cell| obstacle[s * plane + cell] > 0.5)
                .collect();
            ReconstructionDiagnostics {
                grid: grid_from_ids(ids, h, w, &obs_flags),
                confidence: confidence[s * plane..(s + 1) * plane].to_vec(),
                entropy: entropy[s * plane..(s + 1) * plane].to_vec(),
                reveal_step: reveal_step[s * plane..(s + 1) * plane].to_vec(),
            }
        })
        .collect()
}

/// Run the denoiser once and return per-channel softmax probabilities on the
/// host: `probs[c]` is a `[n, K_c, H, W]` flat `Vec<f32>`.
fn predict<B: Backend>(
    model: &Denoiser<B>,
    cur: &[i32],
    obstacle: &[f32],
    [n, h, w]: [usize; 3],
    t_val: f64,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    let tokens = Tensor::<B, 4, Int>::from_data(
        TensorData::new(cur.to_vec(), [n, N_CHANNELS, h, w]),
        device,
    );
    let obstacle =
        Tensor::<B, 4>::from_data(TensorData::new(obstacle.to_vec(), [n, 1, h, w]), device);
    let t = Tensor::<B, 1>::from_data(TensorData::new(vec![t_val as f32; n], [n]), device);

    let logits = model.forward(tokens, obstacle, t);
    logits
        .into_iter()
        .map(|l| softmax(l, 1).to_data().convert::<f32>().into_vec().unwrap())
        .collect()
}

/// Decode a single cell: pick a category per channel (argmax or temperature
/// sample) and return `(confidence_score, ids)`. The score is the summed log-prob
/// of the chosen categories — how sure the model is about this whole cell.
#[derive(Clone, Debug)]
struct CellPrediction {
    score: f64,
    ids: [usize; N_CHANNELS],
    confidence: f32,
    entropy: f32,
}

fn decode_cell(
    probs: &[Vec<f32>],
    s: usize,
    cell: usize,
    plane: usize,
    cfg: &SampleConfig,
    rng: &mut ChaCha8Rng,
) -> CellPrediction {
    let mut ids = [0usize; N_CHANNELS];
    let mut score = 0.0f64;
    let mut confidence = 0.0f64;
    let mut entropy = 0.0f64;
    for c in 0..N_CHANNELS {
        let k = VOCAB[c];
        // probs[c] is [n, K, plane]; index (s, j, cell).
        let p = |j: usize| probs[c][(s * k + j) * plane + cell] as f64;
        let chosen = if cfg.temperature > 0.0 {
            sample_index(k, cfg.temperature, &p, rng)
        } else {
            argmax_index(k, &p)
        };
        ids[c] = chosen;
        score += (p(chosen).max(1e-9)).ln();
        confidence += p(chosen);
        let channel_entropy: f64 = (0..k)
            .map(|j| {
                let probability = p(j).max(1e-12);
                -probability * probability.ln()
            })
            .sum::<f64>()
            / (k as f64).ln();
        entropy += channel_entropy;
    }
    CellPrediction {
        score,
        ids,
        confidence: (confidence / N_CHANNELS as f64) as f32,
        entropy: (entropy / N_CHANNELS as f64) as f32,
    }
}

fn argmax_index(k: usize, p: &impl Fn(usize) -> f64) -> usize {
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for j in 0..k {
        let v = p(j);
        if v > best_v {
            best_v = v;
            best = j;
        }
    }
    best
}

/// Temperature-scaled categorical draw from probabilities `p(0..k)`.
fn sample_index(k: usize, temp: f64, p: &impl Fn(usize) -> f64, rng: &mut ChaCha8Rng) -> usize {
    // Re-normalize p^(1/temp).
    let inv = 1.0 / temp;
    let weights: Vec<f64> = (0..k).map(|j| p(j).max(1e-9).powf(inv)).collect();
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (j, &wgt) in weights.iter().enumerate() {
        r -= wgt;
        if r <= 0.0 {
            return j;
        }
    }
    k - 1
}

fn to_host(partials: &[Grid], observed: &[Vec<bool>]) -> HostInputs {
    let (h, w) = (partials[0].height, partials[0].width);
    let plane = h * w;
    let n = partials.len();
    let mut tokens = vec![0i32; n * N_CHANNELS * plane];
    let mut obstacle = vec![0f32; n * plane];
    let mut obs = vec![false; n * plane];
    for (s, g) in partials.iter().enumerate() {
        assert_eq!((g.height, g.width), (h, w), "ragged batch");
        for cell in 0..plane {
            let (x, y) = (cell % w, cell / w);
            let c = g.get(x, y);
            tokens[(s * N_CHANNELS) * plane + cell] = c.entity as i32;
            tokens[(s * N_CHANNELS + 1) * plane + cell] = c.direction as i32;
            tokens[(s * N_CHANNELS + 2) * plane + cell] = c.item as i32;
            tokens[(s * N_CHANNELS + 3) * plane + cell] = c.misc as i32;
            obstacle[s * plane + cell] = if g.is_obstacle(x, y) { 1.0 } else { 0.0 };
            obs[s * plane + cell] = observed[s][cell];
        }
    }
    HostInputs {
        n,
        h,
        w,
        tokens,
        obstacle,
        observed: obs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::factory_gen::{generate, LessonKind};
    use crate::model::DenoiserConfig;

    #[test]
    fn reconstruct_preserves_observed_and_wellformed_shape() {
        type B = CpuBackend;
        let device = Default::default();
        let s = generate(LessonKind::MoveOneItem, 11, 4).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (partial, observed) = s.blank(None, &mut rng);

        let model = DenoiserConfig::new()
            .with_hidden(16)
            .with_blocks(2)
            .init::<B>(&device);
        let out = reconstruct(
            &model,
            std::slice::from_ref(&partial),
            std::slice::from_ref(&observed),
            &SampleConfig {
                steps: 6,
                ..Default::default()
            },
            &device,
        );
        assert_eq!(out.len(), 1);
        let g = &out[0];
        assert_eq!((g.height, g.width), (11, 11));
        // Observed cells must be preserved exactly (inpainting invariant).
        for (cell, &o) in observed.iter().enumerate() {
            if o {
                let (x, y) = (cell % 11, cell / 11);
                assert_eq!(g.get(x, y), s.solution.get(x, y), "observed cell changed");
            }
        }
    }

    #[test]
    fn diagnostics_cover_every_cell_with_bounded_uncertainty() {
        type B = CpuBackend;
        let device = Default::default();
        let sample = generate(LessonKind::MoveOneItem, 7, 9).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (partial, observed) = sample.blank(None, &mut rng);
        let model = DenoiserConfig::new()
            .with_hidden(8)
            .with_blocks(1)
            .init::<B>(&device);
        let diagnostics = reconstruct_with_diagnostics(
            &model,
            &[partial],
            std::slice::from_ref(&observed),
            &SampleConfig {
                steps: 3,
                ..Default::default()
            },
            &device,
        );
        let diagnostics = &diagnostics[0];
        assert_eq!(diagnostics.confidence.len(), 49);
        assert_eq!(diagnostics.entropy.len(), 49);
        assert_eq!(diagnostics.reveal_step.len(), 49);
        assert!(diagnostics
            .confidence
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v)));
        assert!(diagnostics
            .entropy
            .iter()
            .all(|&v| (0.0..=1.000_001).contains(&v)));
        for (i, &is_observed) in observed.iter().enumerate() {
            assert_eq!(diagnostics.reveal_step[i] == 0, is_observed);
        }
    }
}
