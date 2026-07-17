//! The multi-tile bug, side by side with its fix.
//!
//! Run: `cargo run --example overlap_check`
//!
//! `assembler_line` used to be `S i a i K` on a single row, the assembler taking
//! one cell "simplified from the reference's 3×3 to keep the first model
//! tractable". Nothing in this repo disagreed — every cell was individually
//! well-formed and the simulator scored the factory as perfectly functional.
//!
//! But `blueprint.rs` has always exported a *real* `assembling-machine-1`, which
//! is 3×3 in Factorio and always was. So the machine we anchored at one cell
//! swallowed the two inserters standing beside it, and every blueprint this
//! lesson ever emitted was rejected on import. The only party who could see the
//! problem was Factorio, and nothing here ever asked Factorio.
//!
//! This prints both layouts through the exporter and checks each one the way the
//! game does: by asking whether any two prototypes want the same ground.
//!
//! The regression guard lives in `tests/blueprint_export.rs`; this example is
//! here to *show* it.

use diffusion_factorio::blueprint::grid_to_blueprint;
use diffusion_factorio::factory_gen::{generate, Canvas, LessonKind};
use diffusion_factorio::textual::render;
use diffusion_factorio::world::{Cell, Direction, Entity, Grid, Item};

/// The footprint Factorio enforces for the prototype we export, by name. Not
/// read from `Entity::footprint`: the whole point is to check the export against
/// the game's sizes rather than against our own opinion of them.
fn prototype_size(name: &str) -> (f64, f64) {
    match name {
        "assembling-machine-1" => (3.0, 3.0),
        "splitter" => (2.0, 1.0),
        _ => (1.0, 1.0),
    }
}

/// Count every pair of exported entities that wants the same ground.
fn collisions(grid: &Grid, label: &str) -> usize {
    println!("{label}:");
    println!("{}", render(grid));
    println!(
        "  world model calls this buildable: {}",
        grid.is_consistent()
    );

    let bp = grid_to_blueprint(grid, label).expect("export");
    let boxes: Vec<_> = bp
        .blueprint
        .entities
        .iter()
        .map(|e| {
            let (w, h) = prototype_size(&e.name);
            let (x, y) = (e.position.x, e.position.y);
            (&e.name, x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)
        })
        .collect();

    let mut found = 0;
    for (i, a) in boxes.iter().enumerate() {
        for b in &boxes[i + 1..] {
            if a.1 < b.3 && b.1 < a.3 && a.2 < b.4 && b.2 < a.4 {
                found += 1;
                println!("  COLLISION: {} and {} occupy the same tiles", a.0, b.0);
            }
        }
    }
    println!("  Factorio would reject this blueprint: {}\n", found > 0);
    found
}

/// `S i a i K` on one row: what the lesson used to generate.
fn one_cell_assembler_line() -> Grid {
    let mut grid = Grid::new(5, 1);
    grid.set(
        0,
        0,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    for x in [1, 3] {
        grid.set(
            x,
            0,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
    }
    grid.set(
        2,
        0,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::IronGear,
            ..Default::default()
        },
    );
    grid.set(
        4,
        0,
        Cell {
            entity: Entity::Sink,
            item: Item::IronGear,
            ..Default::default()
        },
    );
    grid
}

fn main() {
    let before = collisions(&one_cell_assembler_line(), "the old 1x1 assembler line");

    let sample = generate(LessonKind::AssemblerLine, Canvas::square(11), 7).expect("gen");
    let after = collisions(&sample.solution, "what the lesson generates now");

    println!("old layout: {before} collision(s); current lesson: {after}");
    assert!(before > 0, "the bug must still be demonstrable");
    assert_eq!(after, 0, "and the current lesson must not have it");
}
