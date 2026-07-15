//! Validation metrics. Three axes, in increasing order of how much they tell
//! you and how much they cost:
//!   * cheap per-cell / per-channel accuracy for fast signal;
//!   * a task-level, simulator-grounded check (does the reconstructed factory
//!     route items at all?) — binary, and saturated: at `functional=0.99` it has
//!     nothing left to say; and
//!   * **graded throughput** (how *fast* does it route them?), which is what
//!     separates two working factories and is the analogue of the reference's
//!     normalized-throughput rollout.
//!
//! The ratio of the reconstruction's rate to the *generator's own answer* is the
//! headline number: 1.0 means the model matched the curriculum's solution, and
//! above 1.0 means it found a better factory than the one it was taught.

use crate::sim::item_reaches_sink;
use crate::throughput;
use crate::world::{Channel, Grid, N_CHANNELS};
use serde::{Deserialize, Serialize};

const CHANNELS: [Channel; N_CHANNELS] = [
    Channel::Entity,
    Channel::Direction,
    Channel::Item,
    Channel::Misc,
];

/// Aggregated reconstruction metrics over a validation set.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReconReport {
    pub n_factories: usize,
    /// Masked cells scored (denominator for per-channel accuracy).
    pub masked_cells: usize,
    /// Per-channel correct counts on masked cells.
    pub channel_correct: [usize; N_CHANNELS],
    /// Factories whose masked cells were reconstructed exactly (all channels).
    pub exact: usize,
    /// Reconstructions that are channel-consistent (well-formed).
    pub consistent: usize,
    /// Reconstructions that are functional (item reaches a sink).
    pub functional: usize,
    /// Of the originals, how many were functional (upper bound / sanity).
    pub original_functional: usize,
    /// Summed items/second delivered by the reconstructions.
    pub throughput: f64,
    /// Summed items/second delivered by the generator's own answers.
    pub original_throughput: f64,
    /// Summed per-task `recon / original` rate ratios, over gradeable tasks.
    pub throughput_ratio: f64,
    /// Tasks whose original answer delivers anything at all — the denominator
    /// for the ratio. A task nobody can score tells us nothing about the model.
    pub gradeable: usize,
    /// Reconstructions that deliver *more* than the answer they were taught.
    /// This is the only metric here that can report a model out-building its
    /// curriculum, so it is the one to watch.
    pub beat_original: usize,
}

impl ReconReport {
    pub fn channel_acc(&self, c: usize) -> f64 {
        if self.masked_cells == 0 {
            0.0
        } else {
            self.channel_correct[c] as f64 / self.masked_cells as f64
        }
    }
    pub fn exact_rate(&self) -> f64 {
        rate(self.exact, self.n_factories)
    }
    pub fn functional_rate(&self) -> f64 {
        rate(self.functional, self.n_factories)
    }
    pub fn consistent_rate(&self) -> f64 {
        rate(self.consistent, self.n_factories)
    }
    /// Mean items/second delivered per reconstructed factory.
    pub fn throughput_mean(&self) -> f64 {
        div(self.throughput, self.n_factories)
    }
    /// Mean items/second delivered by the *generator's own* answers — the
    /// baseline [`Self::throughput_ratio_mean`] divides by. Reported alongside
    /// the ratio because the ratio alone cannot say whether it moved because
    /// the model improved or because the tasks got easier.
    pub fn original_throughput_mean(&self) -> f64 {
        div(self.original_throughput, self.n_factories)
    }
    /// Mean fraction of the generator's own delivered rate that the model
    /// achieved. 1.0 = matched the taught answer; >1.0 = beat it.
    pub fn throughput_ratio_mean(&self) -> f64 {
        div(self.throughput_ratio, self.gradeable)
    }
}

fn rate(num: usize, den: usize) -> f64 {
    div(num as f64, den)
}

fn div(num: f64, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num / den as f64
    }
}

/// Score reconstructions against originals over the masked (unobserved) cells.
pub fn reconstruction_report(
    originals: &[Grid],
    reconstructed: &[Grid],
    observed: &[Vec<bool>],
) -> ReconReport {
    assert_eq!(originals.len(), reconstructed.len());
    assert_eq!(originals.len(), observed.len());
    let mut r = ReconReport {
        n_factories: originals.len(),
        ..Default::default()
    };

    for ((orig, recon), obs) in originals.iter().zip(reconstructed).zip(observed) {
        let mut all_correct = true;
        for (i, &observed_cell) in obs.iter().enumerate().take(orig.len()) {
            if observed_cell {
                continue; // only score masked cells
            }
            r.masked_cells += 1;
            let (oc, rc) = (orig.cells[i], recon.cells[i]);
            let mut cell_correct = true;
            for (ci, ch) in CHANNELS.iter().enumerate() {
                if oc.channel_id(*ch) == rc.channel_id(*ch) {
                    r.channel_correct[ci] += 1;
                } else {
                    cell_correct = false;
                }
            }
            all_correct &= cell_correct;
        }
        if all_correct {
            r.exact += 1;
        }
        if recon.is_consistent() {
            r.consistent += 1;
        }
        if item_reaches_sink(recon) {
            r.functional += 1;
        }
        if item_reaches_sink(orig) {
            r.original_functional += 1;
        }

        // How fast, not just whether. Rates are absolute items/s and are not
        // comparable across tasks -- a gear line and a cable line have different
        // ceilings -- so the ratio against the generator's own answer is what
        // gets averaged.
        let (recon_rate, orig_rate) = (throughput::score(recon), throughput::score(orig));
        r.throughput += recon_rate;
        r.original_throughput += orig_rate;
        if orig_rate > 0.0 {
            r.gradeable += 1;
            r.throughput_ratio += recon_rate / orig_rate;
            if recon_rate > orig_rate {
                r.beat_original += 1;
            }
        }
    }
    r
}

impl std::fmt::Display for ReconReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} | exact={:.3} functional={:.3} (orig_fn={}) consistent={:.3} | thput={:.3}/s ratio={:.3} beat={} | acc[entity={:.3} dir={:.3} item={:.3} misc={:.3}]",
            self.n_factories,
            self.exact_rate(),
            self.functional_rate(),
            self.original_functional,
            self.consistent_rate(),
            self.throughput_mean(),
            self.throughput_ratio_mean(),
            self.beat_original,
            self.channel_acc(0),
            self.channel_acc(1),
            self.channel_acc(2),
            self.channel_acc(3),
        )
    }
}
