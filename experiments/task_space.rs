//! Measure *how much task there is to learn* in the current curriculum.
//!
//! The 5,000-step GPU run reported `exact=1.000 functional=1.000` from step
//! ~3,000 onward. Saturation like that has two possible causes: the model
//! genuinely generalizes, or the task is small enough to memorize. This
//! experiment distinguishes them by measuring the curriculum itself — no model
//! involved.
//!
//! Run: `cargo run --release --example task_space`

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::textual::render;
use diffusion_factorio::world::{Cell, Entity, Grid, Item};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SIZE: usize = 11;
const SEEDS: u64 = 200_000;

/// Canonical key for a full factory. Uses every channel (the ASCII view is
/// glyph-only and would silently collapse recipes/items together).
fn factory_key(g: &Grid) -> String {
    let mut s = String::new();
    for (i, c) in g.cells.iter().enumerate() {
        if !c.is_empty() {
            s.push_str(&format!(
                "{i}:{}:{}:{}:{};",
                c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
            ));
        }
    }
    for y in 0..g.height {
        for x in 0..g.width {
            if g.is_obstacle(x, y) {
                s.push_str(&format!("#{x},{y};"));
            }
        }
    }
    s
}

/// Canonical key for what the model actually *sees* at validation time: the
/// scaffold that stays observed, plus the obstacle plane. Everything else is
/// masked and must be inpainted.
fn conditioning_key(g: &Grid, observed: &[bool]) -> String {
    let mut s = String::new();
    for (i, &obs) in observed.iter().enumerate() {
        if obs {
            let c = g.cells[i];
            s.push_str(&format!(
                "{i}:{}:{}:{}:{};",
                c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
            ));
        }
    }
    for y in 0..g.height {
        for x in 0..g.width {
            if g.is_obstacle(x, y) {
                s.push_str(&format!("#{x},{y};"));
            }
        }
    }
    s
}

/// Canonical key for the answer the model must produce on the masked cells.
fn answer_key(g: &Grid, observed: &[bool]) -> String {
    let mut s = String::new();
    for (i, &obs) in observed.iter().enumerate() {
        if !obs {
            let c = g.cells[i];
            s.push_str(&format!(
                "{i}:{}:{}:{}:{};",
                c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
            ));
        }
    }
    s
}

fn main() {
    println!("=== 1. How many distinct factories can each lesson family produce? ===");
    println!("(size {SIZE}, {SEEDS} distinct generator seeds per family)\n");

    let mut totals = Vec::new();
    for &kind in LessonKind::all() {
        let t0 = Instant::now();
        let mut factories = HashSet::new();
        let mut generated = 0usize;
        // (conditioning -> set of distinct answers) proves whether the label is a
        // deterministic function of what the model conditions on.
        let mut answers_per_context: HashMap<String, HashSet<String>> = HashMap::new();

        for seed in 0..SEEDS {
            let Some(sample) = generate(kind, SIZE, seed) else {
                continue;
            };
            generated += 1;
            factories.insert(factory_key(&sample.solution));

            // Reproduce exactly what train.rs validation does: blank every
            // removable cell, keep the protected scaffold observed.
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (_partial, observed) = sample.blank(None, &mut rng);
            answers_per_context
                .entry(conditioning_key(&sample.solution, &observed))
                .or_default()
                .insert(answer_key(&sample.solution, &observed));
        }

        let ambiguous = answers_per_context
            .values()
            .filter(|answers| answers.len() > 1)
            .count();
        let secs = t0.elapsed().as_secs_f64();
        // The 5,000-step run drew 5,000 * 32 = 160,000 samples uniformly over the
        // 4 families feasible at size 11, so ~40,000 landed in each.
        let draws_per_family = 160_000.0 / LessonKind::all().len() as f64;
        let repeats = draws_per_family / answers_per_context.len() as f64;
        println!(
            "{:<22} distinct factories: {:>6} | distinct tasks: {:>6} | ambiguous tasks: {ambiguous}",
            kind.name(),
            factories.len(),
            answers_per_context.len(),
        );
        println!(
            "{:<22}   {generated} seeds ok, {:.0} gen/s | a 5k-step run sees each task ~{repeats:.1}x",
            "",
            generated as f64 / secs,
        );
        totals.push((kind.name(), answers_per_context.len(), repeats));
    }

    println!("\n  A 'task' = one distinct conditioning the model is asked to complete.");
    println!("  Families small enough to memorize outright (each task seen many times):");
    for (name, tasks, repeats) in totals.iter().filter(|(_, _, r)| *r > 10.0) {
        println!("    {name:<22} {tasks:>6} tasks, seen ~{repeats:.0}x each");
    }
    println!("  Families too large to memorize (each task seen ~once => real generalization):");
    for (name, tasks, repeats) in totals.iter().filter(|(_, _, r)| *r <= 10.0) {
        println!("    {name:<22} {tasks:>6}+ tasks, seen ~{repeats:.1}x each");
    }

    println!("\n=== 2. Is `exact` a fair metric? (is the target label unique?) ===");
    println!("Every conditioning above maps to exactly one answer => the label is a");
    println!("deterministic function of the input. `exact=1.0` is therefore reachable,");
    println!("and since a correct exact match is always functional, functional==exact.");
    println!("This is a property of the *data*, not evidence of factory-design skill.\n");

    println!("=== 3. Is the `functional` metric item-aware? ===");
    // Source provides iron, sink accepts copper, belts connect them.
    let mut g = Grid::new(5, 1);
    g.set(
        0,
        0,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    g.set(1, 0, Cell::belt(diffusion_factorio::world::Direction::East));
    g.set(2, 0, Cell::belt(diffusion_factorio::world::Direction::East));
    g.set(3, 0, Cell::belt(diffusion_factorio::world::Direction::East));
    g.set(
        4,
        0,
        Cell {
            entity: Entity::Sink,
            item: Item::CopperCable,
            ..Default::default()
        },
    );
    println!("  factory: {}", render(&g).trim_end());
    println!("  source provides IronPlate, sink accepts CopperCable (impossible).");
    println!("  item_reaches_sink() = {}", item_reaches_sink(&g));
    println!("  => the metric is purely topological; it never compares items.");
    println!("     The reference guards against exactly this (throughput.rs:205-226,");
    println!("     'sinks only score their configured item') to stop the agent");
    println!("     routing raw input straight to the sink.");
}
