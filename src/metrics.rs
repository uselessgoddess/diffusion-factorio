//! Validation metrics. Two axes, mirroring the reference's philosophy:
//!   * cheap per-cell / per-channel accuracy for fast signal, and
//!   * a task-level, simulator-grounded metric (does the reconstructed factory
//!     actually route items?) — the analogue of their normalized-throughput
//!     rollout, which is what really tells you the model learned something.

use crate::sim::item_reaches_sink;
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
}

fn rate(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
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
    }
    r
}

impl std::fmt::Display for ReconReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} | exact={:.3} functional={:.3} (orig_fn={}) consistent={:.3} | acc[entity={:.3} dir={:.3} item={:.3} misc={:.3}]",
            self.n_factories,
            self.exact_rate(),
            self.functional_rate(),
            self.original_functional,
            self.consistent_rate(),
            self.channel_acc(0),
            self.channel_acc(1),
            self.channel_acc(2),
            self.channel_acc(3),
        )
    }
}
