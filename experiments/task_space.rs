//! Measure *how much task there is to learn* in the current curriculum.
//!
//! The 5,000-step GPU run reported `exact=1.000 functional=1.000` from step
//! ~3,000 onward. Saturation like that has two possible causes: the model
//! genuinely generalizes, or the task is small enough to memorize. This
//! experiment distinguishes them by measuring the curriculum itself — no model
//! involved.
//!
//! Run: `cargo run --release --example task_space [SIZE]`
//!
//! `SIZE` defaults to [`train::TrainConfig`]'s grid size. Pass another to ask
//! the question that answer raised — how much task space a bigger board buys.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::textual::render;
use diffusion_factorio::world::{Cell, Entity, Grid, Item};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

const DEFAULT_SIZE: usize = 11;
const SEEDS: u64 = 200_000;

/// What one lesson family measured out to.
struct Family {
    name: &'static str,
    tasks: usize,
    repeats: f64,
    ambiguous: usize,
    factories: usize,
    /// [`factories`](Self::factories) with translations collapsed.
    shapes: usize,
    /// Distinct *answers* modulo translation — see [`answer_shape_key`].
    answer_shapes: usize,
}

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

/// Canonical key for a factory *modulo translation*: the same layout slid to
/// another corner of the board collapses onto one key.
///
/// This is the honest denominator. [`factory_key`] counts a template at every
/// offset it fits, and those counts are what `docs/ROADMAP.md` reads as "task
/// space" — but the denoiser is a stack of `same`-padded convolutions
/// (`model.rs`, and see `one_set_of_weights_runs_at_any_grid_size`), so
/// translation is the one variation it is equivariant to by construction. A
/// count that grows only because the board got wider is counting free lunches.
fn translation_invariant_key(g: &Grid) -> String {
    let occupied: Vec<(usize, usize, Cell)> = (0..g.height)
        .flat_map(|y| (0..g.width).map(move |x| (x, y)))
        .filter(|&(x, y)| !g.cells[y * g.width + x].is_empty() || g.is_obstacle(x, y))
        .map(|(x, y)| (x, y, g.cells[y * g.width + x]))
        .collect();
    let Some(min_x) = occupied.iter().map(|&(x, _, _)| x).min() else {
        return String::new();
    };
    let min_y = occupied.iter().map(|&(_, y, _)| y).min().unwrap();

    let mut s = String::new();
    for (x, y, c) in occupied {
        let (dx, dy) = (x - min_x, y - min_y);
        if !c.is_empty() {
            s.push_str(&format!(
                "{dx},{dy}:{}:{}:{}:{};",
                c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
            ));
        }
        if g.is_obstacle(x, y) {
            s.push_str(&format!("#{dx},{dy};"));
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

/// Canonical key for the *answer alone*, modulo translation: the cells the model
/// has to fill in, normalized to their own bounding box, with the world they sit
/// in thrown away.
///
/// [`translation_invariant_key`] keys the whole board, obstacles included, and
/// that makes it gameable in exactly one way: scatter random obstacles, and every
/// sample gets a distinct key whether or not the answer ever changes. It would
/// have scored a lesson that generates noise the label ignores as infinitely
/// varied. Since the fix for the templated families *is* to scatter obstacles
/// ([`LessonKind::AssemblerChaos`]), that is the one number it must not be
/// allowed to inflate, so this counts what the model is actually asked to
/// produce.
///
/// Distinctness here is still not sufficient — two belt runs differing in one
/// tile are two keys and nearly one lesson — but it is necessary, and it is what
/// separates a family that varies its answers from one that varies its wallpaper.
fn answer_shape_key(g: &Grid, observed: &[bool]) -> String {
    let cells: Vec<(usize, usize, Cell)> = observed
        .iter()
        .enumerate()
        .filter(|&(_, &obs)| !obs)
        .map(|(i, _)| (i % g.width, i / g.width, g.cells[i]))
        .filter(|(_, _, c)| !c.is_empty())
        .collect();
    let Some(min_x) = cells.iter().map(|&(x, _, _)| x).min() else {
        return String::new();
    };
    let min_y = cells.iter().map(|&(_, y, _)| y).min().unwrap();

    let mut s = String::new();
    for (x, y, c) in cells {
        s.push_str(&format!(
            "{},{}:{}:{}:{}:{};",
            x - min_x,
            y - min_y,
            c.entity as u8,
            c.direction as u8,
            c.item as u8,
            c.misc as u8
        ));
    }
    s
}

fn main() {
    let size: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(DEFAULT_SIZE);

    let feasible = LessonKind::all()
        .iter()
        .filter(|k| size >= k.min_size())
        .count();

    println!("=== 1. How many distinct factories can each lesson family produce? ===");
    println!("(size {size}, {SEEDS} distinct generator seeds per family)");
    println!(
        "({feasible} of {} families fit at this size)\n",
        LessonKind::all().len()
    );

    let mut totals = Vec::new();
    for &kind in LessonKind::all().iter().filter(|k| size >= k.min_size()) {
        let t0 = Instant::now();
        let mut factories = HashSet::new();
        let mut shapes = HashSet::new();
        let mut answer_shapes = HashSet::new();
        let mut generated = 0usize;
        // (conditioning -> set of distinct answers) proves whether the label is a
        // deterministic function of what the model conditions on.
        let mut answers_per_context: HashMap<String, HashSet<String>> = HashMap::new();

        for seed in 0..SEEDS {
            let Some(sample) = generate(kind, size, seed) else {
                continue;
            };
            generated += 1;
            factories.insert(factory_key(&sample.solution));
            shapes.insert(translation_invariant_key(&sample.solution));

            // Reproduce exactly what train.rs validation does: blank every
            // removable cell, keep the protected scaffold observed.
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (_partial, observed) = sample.blank(None, &mut rng);
            answer_shapes.insert(answer_shape_key(&sample.solution, &observed));
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
        // A 5,000-step run draws 5,000 * 32 = 160,000 samples uniformly over the
        // families that *fit* — train.rs skips the rest, so a size below 11 (a
        // circuit line is 11 wide) spreads the same budget over fewer families
        // and makes each of them look more memorized than it is.
        let draws_per_family = 160_000.0 / feasible as f64;
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
        println!(
            "{:<22}   distinct *shapes* (same layout at another offset collapsed): {:>6}",
            "",
            shapes.len(),
        );
        totals.push(Family {
            name: kind.name(),
            tasks: answers_per_context.len(),
            repeats,
            ambiguous,
            factories: factories.len(),
            shapes: shapes.len(),
            answer_shapes: answer_shapes.len(),
        });
    }

    println!("\n  A 'task' = one distinct conditioning the model is asked to complete.");
    println!("  Families small enough to memorize outright (each task seen many times):");
    for f in totals.iter().filter(|f| f.repeats > 10.0) {
        println!(
            "    {:<22} {:>6} tasks, seen ~{:.0}x each",
            f.name, f.tasks, f.repeats
        );
    }
    println!("  Families too large to memorize (each task seen ~once => real generalization):");
    for f in totals.iter().filter(|f| f.repeats <= 10.0) {
        println!(
            "    {:<22} {:>6}+ tasks, seen ~{:.1}x each",
            f.name, f.tasks, f.repeats
        );
    }

    println!("\n=== 1b. How much of that is structure, and how much is sliding? ===");
    println!("  The denoiser is `same`-padded convolutions end to end, so a layout");
    println!("  moved to another offset is the one variation it generalizes over for");
    println!("  free. Collapsing translations is therefore the honest count of what");
    println!("  a family actually teaches:\n");
    println!(
        "  {:<22} {:>10} {:>8} {:>13} {:>9}",
        "family", "factories", "shapes", "of which new", "answers"
    );
    for f in &totals {
        let slide = f.factories as f64 / f.shapes.max(1) as f64;
        println!(
            "  {:<22} {:>10} {:>8} {:>11.0}x translation {:>7}",
            f.name, f.factories, f.shapes, slide, f.answer_shapes
        );
    }
    println!("\n  A family whose `shapes` count stays flat as SIZE grows is not gaining");
    println!("  task space on a bigger board -- it is gaining offsets. Compare two");
    println!("  sizes to see which of these numbers actually move.");
    println!("\n  `answers` is the column to trust. `shapes` keys the whole board,");
    println!("  obstacles and all, so any family that scatters obstacles scores a");
    println!("  perfect count for free -- whether or not the obstacles change the");
    println!("  label. `answers` keys only the cells the model must fill in, modulo");
    println!("  translation, so it is the one number a chaos family cannot fake.");

    println!("\n=== 2. Is `exact` a fair metric? (is the target label unique?) ===");
    let (rigid, ambiguous): (Vec<_>, Vec<_>) = totals.iter().partition(|f| f.ambiguous == 0);
    for f in &rigid {
        println!(
            "  {:<22} every one of its {} tasks has exactly 1 answer",
            f.name, f.tasks
        );
    }
    println!("For these the label is a deterministic function of the input. `exact=1.0`");
    println!("is reachable, and since a correct exact match is always functional,");
    println!("functional==exact. That is a property of the *data*, not evidence of");
    println!("factory-design skill -- and it is why the 5,000-step run's two headline");
    println!("metrics moved as one: there was only ever one answer, so getting it right");
    println!("and getting it working were the same event.\n");
    for f in &ambiguous {
        println!(
            "  {:<22} {} of its {} tasks admit more than one answer",
            f.name, f.ambiguous, f.tasks
        );
    }
    if ambiguous.is_empty() {
        println!("  (none -- nothing here can teach the model to choose)");
    } else {
        println!("For these `exact` is no longer the right question: the model is asked");
        println!("for *a* working factory, not for the one the generator happened to roll,");
        println!("so `exact` is capped below 1.0 by construction while `functional` and");
        println!("throughput are not. This is what gives Best-of-N a choice to make and a");
        println!("policy gradient something to push on.");
    }
    println!();

    println!("=== 3. Is the `functional` metric item-aware? ===");
    // Belt raw iron straight into a sink that wants gears: well-connected, but
    // it can never deliver. A purely topological check calls this functional.
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
            item: Item::IronGear,
            ..Default::default()
        },
    );
    println!("  factory: {}", render(&g).trim_end());
    println!("  source provides IronPlate, sink accepts IronGear, no assembler.");
    println!("  item_reaches_sink() = {}", item_reaches_sink(&g));
    println!("  => must be false: the flow carries the item, and a gear sink");
    println!("     rejects raw plates. Before the fix this returned true, which");
    println!("     rewarded belting raw input straight to the sink instead of");
    println!("     building the assembler. The reference guards the same hole");
    println!("     (throughput.rs:205-226, 'sinks only score their configured item').");
    assert!(
        !item_reaches_sink(&g),
        "functional check regressed to being item-blind"
    );
}
