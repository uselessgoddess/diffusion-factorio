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
use crate::world;
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

/// How many of a task's cells are still masked after `done` of `steps` rounds.
///
/// The cosine schedule: the model commits few cells early, when almost nothing is
/// decided and its predictions are worth little, and most of them late, once the
/// context is dense. `masked0 * cos(π/2 · done/steps)`, hitting exactly zero on
/// the final round.
///
/// The shape has a cost the issue noticed: the reveal *rate* is the derivative of
/// the cosine, so it is steepest at the end. At `steps = 12` the last three rounds
/// commit 38% of the grid between them, which is why the animation looks like the
/// factory appears all at once, and why raising `steps` visibly helps — it is not
/// more thinking so much as a gentler final slope. See
/// `experiments/reveal_schedule.rs`.
pub fn still_masked_after(done: usize, steps: usize, masked0: usize) -> usize {
    if done >= steps {
        return 0;
    }
    let frac = (std::f64::consts::FRAC_PI_2 * done as f64 / steps as f64).cos();
    (masked0 as f64 * frac).round() as usize
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
            let should_remain = still_masked_after(step + 1, steps, masked0[s]);
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

/// Decode a single cell: pick the most likely **legal** combination of the four
/// channels (argmax or temperature sample) and return its joint log-probability.
/// The score is how sure the model is about this whole cell, and it is what the
/// reveal order sorts on.
#[derive(Clone, Debug)]
struct CellPrediction {
    score: f64,
    ids: [usize; N_CHANNELS],
    confidence: f32,
    entropy: f32,
}

/// Choose a cell by scoring the 57 legal combinations under the product of the
/// per-channel heads, rather than choosing each channel on its own.
///
/// Picking each channel independently is what put `TransportBelt` next to
/// `Direction::None` in a factory the user could not import. The heads are not
/// wrong when that happens — the entity head correctly reports that a belt is
/// more likely than floor, and the direction head correctly reports that no
/// single heading beats `None` once the belt's mass is split four ways. The
/// mistake is in combining them, because their product ranges over 720
/// combinations and only 57 of those are cells. See [`world::legal_cells`].
///
/// Restricting the argmax to the legal set is the same model, read correctly:
/// it is the maximum-likelihood cell under the network's own distribution given
/// that a cell must be buildable. It cannot be wrong more often than the
/// unconstrained argmax is *right*, because every choice it rules out is one no
/// factory can contain. `Grid::is_consistent` is therefore no longer a
/// validation metric that can fail at inference — it is an invariant.
fn decode_cell(
    probs: &[Vec<f32>],
    s: usize,
    cell: usize,
    plane: usize,
    cfg: &SampleConfig,
    rng: &mut ChaCha8Rng,
) -> CellPrediction {
    // probs[c] is [n, K, plane]; index (s, j, cell).
    let p = |c: usize, j: usize| probs[c][(s * VOCAB[c] + j) * plane + cell] as f64;
    let joint = |ids: &[usize; N_CHANNELS]| -> f64 {
        (0..N_CHANNELS)
            .map(|c| p(c, ids[c]).max(1e-9).ln())
            .sum::<f64>()
    };

    let legal = world::legal_cells();
    let scores: Vec<f64> = legal.iter().map(joint).collect();
    let pick = if cfg.temperature > 0.0 {
        sample_legal(&scores, cfg.temperature, rng)
    } else {
        argmax_legal(&scores)
    };
    let ids = legal[pick];

    // Diagnostics stay per-channel so the viewer's confidence and entropy
    // overlays remain comparable across the two decoders.
    let confidence: f64 = (0..N_CHANNELS).map(|c| p(c, ids[c])).sum::<f64>() / N_CHANNELS as f64;
    let entropy: f64 = (0..N_CHANNELS)
        .map(|c| {
            let k = VOCAB[c];
            (0..k)
                .map(|j| {
                    let probability = p(c, j).max(1e-12);
                    -probability * probability.ln()
                })
                .sum::<f64>()
                / (k as f64).ln()
        })
        .sum::<f64>()
        / N_CHANNELS as f64;

    CellPrediction {
        score: scores[pick],
        ids,
        confidence: confidence as f32,
        entropy: entropy as f32,
    }
}

fn argmax_legal(scores: &[f64]) -> usize {
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (j, &v) in scores.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = j;
        }
    }
    best
}

/// Temperature-scaled categorical draw over the legal cells, from their joint
/// log-probabilities. Softmax is taken in log-space against the maximum so a
/// long-tailed 57-way distribution cannot underflow to all-zero weights — the
/// old per-channel version raised probabilities to `1/temp` directly, which is
/// only safe because each channel had at most eight categories.
fn sample_legal(scores: &[f64], temp: f64, rng: &mut ChaCha8Rng) -> usize {
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| ((s - top) / temp).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (j, &wgt) in weights.iter().enumerate() {
        r -= wgt;
        if r <= 0.0 {
            return j;
        }
    }
    scores.len() - 1
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
    use crate::world::{Cell, Direction, Entity};

    /// Build a one-cell `probs` layout so a decoder can be tested against a
    /// distribution we choose, with no model in the way. `per_channel[c][j]` is
    /// the probability the head for channel `c` gives category `j`.
    fn probs_for(per_channel: [Vec<f32>; N_CHANNELS]) -> Vec<Vec<f32>> {
        per_channel.into_iter().collect()
    }

    /// The schedule has to reach zero on the last round, or the sampler would
    /// hand back a grid with MASK still in it. The `.max(1)` floor at the call
    /// site hides a violation for small grids by forcing a reveal anyway, so it
    /// is stated here where nothing can cover for it.
    #[test]
    fn the_schedule_leaves_nothing_masked_at_the_end() {
        for steps in [1, 2, 4, 12, 24, 64] {
            for masked0 in [1, 7, 115, 1024] {
                assert_eq!(
                    still_masked_after(steps, steps, masked0),
                    0,
                    "steps={steps} masked0={masked0}"
                );
            }
        }
    }

    /// Cells are committed, never un-committed: the count still masked can only
    /// fall. A schedule that rose would ask the loop to reveal a negative number
    /// of cells, which `saturating_sub` would silently read as zero.
    #[test]
    fn the_schedule_never_asks_for_a_cell_back() {
        let (steps, masked0) = (12, 115);
        for done in 0..steps {
            assert!(
                still_masked_after(done + 1, steps, masked0)
                    <= still_masked_after(done, steps, masked0),
                "round {done} masks more than round {}",
                done + 1
            );
        }
        assert_eq!(still_masked_after(0, steps, masked0), masked0);
    }

    /// The concentration the issue saw as "the whole factory appears in the last
    /// 2-3 frames". It is the schedule, not the model: at the default 12 steps the
    /// last three rounds commit over a third of the grid, and doubling the steps
    /// roughly halves that. See `experiments/reveal_schedule.rs`.
    #[test]
    fn most_of_the_grid_is_committed_in_the_last_few_rounds() {
        let masked0 = 115;
        let tail =
            |steps: usize| still_masked_after(steps - 3, steps, masked0) as f64 / masked0 as f64;
        assert!(
            (tail(12) - 0.383).abs() < 0.01,
            "steps=12 tail was {}",
            tail(12)
        );
        assert!(
            (tail(24) - 0.195).abs() < 0.01,
            "steps=24 tail was {}",
            tail(24)
        );
        assert!(tail(24) < tail(12), "more steps must soften the ending");
    }

    /// Per-channel argmax over these four heads, spelled out rather than
    /// trusted. This is what the decoder used to do.
    fn per_channel_argmax(per_channel: &[Vec<f32>; N_CHANNELS]) -> [usize; N_CHANNELS] {
        std::array::from_fn(|c| {
            per_channel[c]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
    }

    /// A belt whose heading is split four ways, against floor. `None` collects
    /// the whole 30% that belongs to floor while each real heading is scored on
    /// its own, so `None` takes the direction argmax on a plurality even though
    /// the entity head says a belt is here by a majority. Per-channel argmax
    /// multiplies the two correct answers together and reports a belt facing
    /// nowhere: `cannot export inconsistent cell`, verbatim.
    ///
    /// Reading the same heads jointly keeps the belt *and* commits it east —
    /// 0.7 × 0.28 beats floor's 0.3 × 0.3. Nothing about the model changed; only
    /// the illegal way of reading it is gone.
    #[test]
    fn a_split_direction_vote_no_longer_builds_a_belt_facing_nowhere() {
        let mut entity = vec![0.0; VOCAB[0]];
        entity[Entity::Empty as usize] = 0.3;
        entity[Entity::TransportBelt as usize] = 0.7;
        let mut direction = vec![0.0; VOCAB[1]];
        direction[Direction::None as usize] = 0.3;
        direction[Direction::East as usize] = 0.28;
        direction[Direction::North as usize] = 0.16;
        direction[Direction::South as usize] = 0.16;
        direction[Direction::West as usize] = 0.1;
        let mut item = vec![0.0; VOCAB[2]];
        item[0] = 1.0;
        let mut misc = vec![0.0; VOCAB[3]];
        misc[0] = 1.0;
        let per_channel = [entity, direction, item, misc];

        let old = per_channel_argmax(&per_channel);
        assert_eq!(old[0], Entity::TransportBelt as usize);
        assert_eq!(old[1], Direction::None as usize);
        assert!(
            !Cell::from_ids(old).unwrap().is_consistent(),
            "the bug being fixed: per-channel argmax builds a belt facing nowhere"
        );

        let probs = probs_for(per_channel);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let got = decode_cell(&probs, 0, 0, 1, &SampleConfig::default(), &mut rng);
        let cell = Cell::from_ids(got.ids).unwrap();
        assert!(cell.is_consistent(), "decoded {cell:?}");
        assert_eq!(cell.entity, Entity::TransportBelt);
        assert_eq!(cell.direction, Direction::East);
    }

    /// When the model is genuinely torn — 55% a belt, but so unsure of the
    /// heading that no single belt cell (0.55 × 0.1375) outscores floor
    /// (0.45 × 0.45) — the joint decoder answers floor rather than inventing a
    /// heading it does not believe, and scores the answer low.
    ///
    /// The low score is the point. Reveal order sorts on it, so an unresolved
    /// cell now waits for its neighbours to commit instead of being drawn early
    /// on the strength of two marginals that were never about the same cell.
    /// Under the old decoder this cell scored log 0.55 + log 0.45 — confident,
    /// and wrong.
    #[test]
    fn a_cell_the_model_cannot_resolve_scores_low_instead_of_guessing() {
        let mut entity = vec![0.0; VOCAB[0]];
        entity[Entity::Empty as usize] = 0.45;
        entity[Entity::TransportBelt as usize] = 0.55;
        let mut direction = vec![0.0; VOCAB[1]];
        direction[Direction::None as usize] = 0.45;
        for d in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            direction[d as usize] = 0.55 / 4.0;
        }
        let mut item = vec![0.0; VOCAB[2]];
        item[0] = 1.0;
        let mut misc = vec![0.0; VOCAB[3]];
        misc[0] = 1.0;

        let probs = probs_for([entity, direction, item, misc]);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let got = decode_cell(&probs, 0, 0, 1, &SampleConfig::default(), &mut rng);
        let cell = Cell::from_ids(got.ids).unwrap();
        assert!(cell.is_consistent(), "decoded {cell:?}");
        assert_eq!(cell.entity, Entity::Empty);
        // ln(0.45 * 0.45) = -1.6, against the -1.09 the old decoder would have
        // reported for its unbuildable belt.
        assert!(got.score < (0.55f64 * 0.45).ln(), "score {}", got.score);
    }

    /// Constraint, not luck: every legal-set draw is a cell, at any temperature
    /// and under any distribution the heads produce.
    #[test]
    fn no_distribution_and_no_temperature_can_decode_an_illegal_cell() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        for trial in 0..2_000 {
            let per_channel: [Vec<f32>; N_CHANNELS] = std::array::from_fn(|c| {
                let raw: Vec<f32> = (0..VOCAB[c]).map(|_| rng.gen::<f32>()).collect();
                let total: f32 = raw.iter().sum();
                raw.into_iter().map(|v| v / total).collect()
            });
            let probs = probs_for(per_channel);
            let cfg = SampleConfig {
                temperature: if trial % 2 == 0 { 0.0 } else { 1.0 },
                ..Default::default()
            };
            let got = decode_cell(&probs, 0, 0, 1, &cfg, &mut rng);
            let cell = Cell::from_ids(got.ids).expect("decoded ids are real categories");
            assert!(cell.is_consistent(), "trial {trial} decoded {cell:?}");
        }
    }

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
