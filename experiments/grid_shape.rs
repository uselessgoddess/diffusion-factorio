//! Does the model care what shape the canvas is?
//!
//! The issue trains on 11x11 and then infers on 13x9, which is the setup the
//! whole request rests on:
//!
//! > Максимально обобщить, чтобы модель понимала что для производства шестерёнок
//! > нужно пластины железные в сборщик любой ценой запихнуть и как-то уместить,
//! > независимо от места на grid.
//!
//! `model::tests::one_set_of_weights_runs_at_any_grid_size` proves the weights
//! *run* at any size — the denoiser is fully convolutional, so nothing throws.
//! It does not prove they *work*, and those are different claims. Every lesson
//! in `factory_gen` builds on `Grid::new(size, size)`, so a square canvas is the
//! only shape the model has ever been shown.
//!
//! The confound to avoid: a wider canvas usually means a longer route, and a
//! longer route is harder for reasons that have nothing to do with shape. So the
//! task here is held *identical* — source and sink a fixed distance apart, both
//! centred — and only the canvas around them changes. 13x9 is the useful
//! comparison: 117 cells against 11x11's 121, so the area is almost exactly
//! matched and the aspect ratio is what differs.
//!
//! ```text
//! cargo run --release --example grid_shape -- <checkpoint>
//! ```

use diffusion_factorio::backend::CpuBackend;
use diffusion_factorio::best_of_n::{best_of_n, usable_score, BestOfNConfig};
use diffusion_factorio::model::Denoiser;
use diffusion_factorio::persist;
use diffusion_factorio::sample::SampleConfig;
use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::world::{Cell, Entity, Grid, Item};
use std::path::PathBuf;

/// Cells between source and sink. Fixed across every canvas: this is the task,
/// and it does not get harder just because the canvas around it changed.
const SPAN: usize = 6;

/// Canvases to compare. The first is what the curriculum trains on; the second
/// is what the issue ran inference on.
const SHAPES: [(usize, usize, &str); 8] = [
    (11, 11, "trained shape"),
    (13, 9, "the issue's shape"),
    (9, 13, "the same, turned"),
    (11, 9, ""),
    (9, 11, ""),
    (13, 13, ""),
    (15, 15, ""),
    (9, 9, ""),
];

/// Pin a source and a sink `SPAN` apart, centred, and observe nothing else.
///
/// `row` shifts the pair off the middle row so the table is not one lucky
/// placement. Returns `None` where the canvas cannot hold the task.
fn task(width: usize, height: usize, row: usize) -> Option<(Grid, Vec<bool>)> {
    if width < SPAN + 2 || row >= height {
        return None;
    }
    let mut grid = Grid::new(width, height);
    let x0 = (width - SPAN) / 2;
    grid.set(
        x0,
        row,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    grid.set(
        x0 + SPAN,
        row,
        Cell {
            entity: Entity::Sink,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    let observed = grid
        .cells
        .iter()
        .map(|c| matches!(c.entity, Entity::Source | Entity::Sink))
        .collect();
    Some((grid, observed))
}

fn main() -> anyhow::Result<()> {
    let checkpoint = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: grid_shape <checkpoint>"))?,
    );
    let device = Default::default();
    let model: Denoiser<CpuBackend> = persist::load(&checkpoint, &device)?;

    let cfg = BestOfNConfig {
        n: 8,
        sample: SampleConfig {
            steps: 12,
            temperature: 0.9,
            seed: 7,
        },
        prefer_compact: true,
    };

    println!(
        "One task -- iron plate from a source to a sink {SPAN} cells east -- posed on\n\
         canvases of different shapes. The task never changes. Only the canvas does.\n\
         Checkpoint: {}\n",
        checkpoint.display()
    );
    println!("      canvas  cells  items/s  buildable  functional  note");

    for (width, height, note) in SHAPES {
        // Every row the pair fits on, so no single placement carries the result.
        let (mut partials, mut observed) = (Vec::new(), Vec::new());
        for row in 0..height {
            if let Some((p, o)) = task(width, height, row) {
                partials.push(p);
                observed.push(o);
            }
        }
        if partials.is_empty() {
            continue;
        }
        let runs = best_of_n(&model, &partials, &observed, &cfg, &device);
        let n = runs.len() as f64;
        let thput: f64 = runs.iter().map(|r| usable_score(&r.best.grid)).sum::<f64>() / n;
        let buildable: f64 = runs
            .iter()
            .map(|r| f64::from(r.best.grid.is_consistent()))
            .sum::<f64>()
            / n;
        let functional: f64 = runs
            .iter()
            .map(|r| f64::from(item_reaches_sink(&r.best.grid)))
            .sum::<f64>()
            / n;
        println!(
            "  {:>4}x{:<4}  {:>5}  {:>7.3}  {:>8.0}%  {:>9.0}%  {note}",
            width,
            height,
            width * height,
            thput,
            100.0 * buildable,
            100.0 * functional,
        );
    }

    println!(
        "\nRead the first two rows against each other. 11x11 and 13x9 are within four\n\
         cells of the same area and hold the identical task, so a gap between them is\n\
         the model reacting to the shape of the canvas rather than to the job."
    );
    Ok(())
}
