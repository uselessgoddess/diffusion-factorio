//! Can a factory feed several machines from *one* line, the way Factorio does?
//!
//! The `ASSEMBLER_BANK` lesson gives each of its three machines a private
//! [`Entity::Source`] ([`factory_gen::BANK_LINES`] of them, one per row). No real
//! factory looks like that. A real one runs a single belt past the machines and
//! either taps it with an inserter per machine, or splits it with a splitter.
//! Both raise throughput per input line, and both are the sort of thing we want
//! the model to *invent* rather than copy.
//!
//! Before asking whether the model can draw those patterns, ask whether the
//! grader can score them — a pattern the simulator calls broken is one best-of-N
//! will always discard and one every loss will punish. So this hand-builds the
//! three layouts and grades them:
//!
//! ```text
//! cargo run --release --example bus_tap
//! ```
//!
//! Part two counts what the lessons actually contain, because a pattern that
//! grades well but never appears in a lesson is one the model has never seen.

use diffusion_factorio::{
    factory_gen::{generate, LessonKind, BANK_LINES},
    sim, textual,
    throughput::{throughput, BELT_RATE},
    world::{Cell, Direction, Entity, Grid, Item},
};

fn anchor(entity: Entity, item: Item) -> Cell {
    Cell {
        entity,
        item,
        ..Default::default()
    }
}

fn inserter(direction: Direction) -> Cell {
    Cell {
        entity: Entity::Inserter,
        direction,
        ..Default::default()
    }
}

fn assembler(recipe: Item) -> Cell {
    Cell {
        entity: Entity::Assembler,
        direction: Direction::East,
        item: recipe,
        ..Default::default()
    }
}

/// Three machines, three sources — the shape `gen_assembler_bank` teaches.
///
/// ```text
///     . . A A A . .
///     S i A A A i K     x3, stacked
///     . . A A A . .
/// ```
fn three_sources(recipe: Item, ingredient: Item) -> Grid {
    let mut g = Grid::new(7, 3 * BANK_LINES);
    for line in 0..BANK_LINES {
        let mid = line * 3 + 1;
        g.set(0, mid, anchor(Entity::Source, ingredient));
        g.set(1, mid, inserter(Direction::East));
        g.set(2, mid - 1, assembler(recipe));
        g.set(5, mid, inserter(Direction::East));
        g.set(6, mid, anchor(Entity::Sink, recipe));
    }
    g
}

/// One source, one belt, three inserters tapping it as it runs past.
///
/// This is the pattern the issue asks for: "манипуляторами забирать с одной
/// линии последовательно" — inserters taking off a single line, in sequence.
///
/// ```text
///     S > > > > > > > > > > > >    the bus
///     . . v . . . v . . . v . .    tap inserters, swinging south
///     . a A A a . A A A . A A A    machines
///     ...
/// ```
fn shared_bus(recipe: Item, ingredient: Item) -> Grid {
    let mut g = Grid::new(13, 7);
    // The bus: a source at the head, belt all the way east.
    g.set(0, 0, anchor(Entity::Source, ingredient));
    for x in 1..13 {
        g.set(x, 0, Cell::belt(Direction::East));
    }
    for ax in [1usize, 5, 9] {
        let tap = ax + 1;
        // Picks up from the belt tile behind it (north) and swings into the
        // machine's top row.
        g.set(tap, 1, inserter(Direction::South));
        g.set(ax, 2, assembler(recipe));
        // Unload the machine's bottom row into a sink of its own.
        g.set(tap, 5, inserter(Direction::South));
        g.set(tap, 6, anchor(Entity::Sink, recipe));
    }
    g
}

/// One source, one belt, a splitter tree dividing it between two machines.
///
/// A splitter is the other way Factorio feeds many machines from one line, and
/// unlike the tap it is a first-class entity in our vocabulary — so if this one
/// scores, the gap is the curriculum's, not the simulator's.
///
/// ```text
///     S > = > i A A A . .
///     . . = > i A A A . .
/// ```
fn shared_bus_splitter(recipe: Item, ingredient: Item) -> Grid {
    let mut g = Grid::new(10, 8);
    g.set(0, 0, anchor(Entity::Source, ingredient));
    g.set(1, 0, Cell::belt(Direction::East));
    // East-facing splitter: 1 wide, 2 tall, anchored at the top tile. Each of
    // its two tiles pushes east independently — one belt in, two belts out.
    g.set(
        2,
        0,
        Cell {
            entity: Entity::Splitter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    // Each branch: belt east, inserter east, machine, inserter east, sink. The
    // branches are 4 rows apart so the 3×3 machines never overlap.
    for row in [0usize, 4] {
        if row != 0 {
            // Route the splitter's lower output down to the second branch.
            for y in 1..=row {
                g.set(3, y, Cell::belt(Direction::South));
            }
        }
        g.set(3, row, Cell::belt(Direction::East));
        g.set(4, row, inserter(Direction::East));
        g.set(5, row, assembler(recipe));
        g.set(8, row, inserter(Direction::East));
        g.set(9, row, anchor(Entity::Sink, recipe));
    }
    g
}

fn grade(label: &str, g: &Grid) {
    let report = throughput(g);
    let legal = g.footprints_are_legal();
    let functional = sim::item_reaches_sink(g);
    println!("=== {label} ===");
    print!("{}", textual::render(g));
    println!(
        "  footprints legal: {legal}    reaches a sink: {functional}    score: {:.3}",
        report.score
    );
    for d in &report.deliveries {
        println!(
            "    sink at {:?} wants {:?}: {:.3}/s",
            d.at, d.item, d.achieved
        );
    }
    println!();
}

/// How many of each entity the lessons actually place, per family.
fn census() {
    println!("=== what the lessons teach (200 seeds each, size 11) ===");
    println!(
        "{:<24} {:>7} {:>7} {:>9} {:>9} {:>9}",
        "lesson", "srcs/f", "sinks/f", "belts/f", "splitters", "inserters"
    );
    for &kind in LessonKind::all() {
        let mut counts = [0usize; Entity::COUNT];
        let mut factories = 0usize;
        for seed in 0..200u64 {
            let Some(s) = generate(kind, 11, seed) else {
                continue;
            };
            factories += 1;
            for c in &s.solution.cells {
                counts[c.entity as usize] += 1;
            }
        }
        if factories == 0 {
            continue;
        }
        let per = |e: Entity| counts[e as usize] as f64 / factories as f64;
        println!(
            "{:<24} {:>7.2} {:>7.2} {:>9.2} {:>9} {:>9.2}",
            kind.name(),
            per(Entity::Source),
            per(Entity::Sink),
            per(Entity::TransportBelt),
            // Not a rate: the total across every factory, because the claim
            // being tested is that it is exactly zero.
            counts[Entity::Splitter as usize],
            per(Entity::Inserter),
        );
    }
    println!();
}

fn main() {
    let recipe = Item::IronGear;
    let ingredient = Item::IronPlate;

    println!(
        "A belt carries {BELT_RATE}/s. One gear costs 2 iron plate, and an \
         assembler-1 runs a 0.5s recipe at speed 0.5.\n"
    );

    grade(
        "three sources (what ASSEMBLER_BANK teaches)",
        &three_sources(recipe, ingredient),
    );
    grade(
        "one shared bus, three sequential taps",
        &shared_bus(recipe, ingredient),
    );
    grade(
        "one shared bus, split by a splitter",
        &shared_bus_splitter(recipe, ingredient),
    );

    census();
}
