//! The issue's own failing layouts, rebuilt and put to the simulator.
//!
//! The issue looks at its first result and asks a direct question:
//!
//! > Хотя вроде как тут всё окей, разве что два конвейера входят в sink.
//! > Или я тут не прав?
//!
//! The answer is that the two belts are innocent and the layout is not "вроде
//! как окей" — it has two independent faults, either of which alone delivers
//! nothing, and neither is the thing that was noticed. This file rebuilds the
//! layout and repairs it one fault at a time, so the claim is a table rather
//! than an opinion.
//!
//! The grid is not read off the ASCII, which cannot show an inserter's facing
//! (`textual::glyph` draws every inserter as `i`) and therefore cannot settle
//! the question. It is transcribed from the blueprint string in the issue —
//! ground truth, since that is what the model actually emitted:
//!
//! ```text
//! 11.5 0.5  transport-belt         dir=4      12.5 0.5  transport-belt  dir=8
//! 12.5 1.5  transport-belt         dir=8      12.5 2.5  transport-belt  dir=8
//! 12.5 3.5  transport-belt         dir=8       0.5 4.5  source(iron-plate)
//!  1.5 4.5 .. 6.5 4.5  transport-belt dir=4   10.5 4.5  inserter        dir=4
//!  8.5 5.5  assembling-machine-1              11.5 4.5  transport-belt  dir=4
//! 12.5 4.5  sink(iron-gear-wheel)
//! ```
//!
//! Two things in that dump are worth staring at. The `assembling-machine-1`
//! carries **no `recipe` field** — `blueprint.rs:127` writes one whenever the
//! cell has a craftable tag, so its absence means the cell had none. And the
//! sink is tagged `iron-gear-wheel`, which is the issue's other complaint:
//!
//! > к тому же она почему-то стремится производить тут gear, хотя я не просил
//!
//! It was asked. The sink demands gears, gears exist only as an assembler
//! recipe, and building one is the only response to that task that could ever
//! score. The model was right about the plan and wrong about the details.
//!
//! Run: `cargo run --release --example issue_examples`

use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::throughput;
use diffusion_factorio::world::{Cell, Direction, Entity, Grid, Item};
use diffusion_factorio::{blueprint, textual};

fn belt(direction: Direction) -> Cell {
    Cell {
        entity: Entity::TransportBelt,
        direction,
        ..Default::default()
    }
}

fn port(entity: Entity, item: Item) -> Cell {
    Cell {
        entity,
        item,
        ..Default::default()
    }
}

/// Example 1, transcribed from the issue's blueprint string.
///
/// The assembler's recipe tag is `Item::None` because the exported blueprint
/// has no recipe field; its direction is East because the export says `dir=4`
/// and [`Cell::is_consistent`] requires a machine to face somewhere.
///
/// That untagged machine is no longer a legal cell, so this function builds one
/// on purpose. The point is to rebuild what the model *emitted*, under the rules
/// it emitted it under — not what would pass today.
fn example_one() -> Grid {
    let mut grid = Grid::new(13, 9);

    // The chain in the top-right corner, fed by nothing, ending in the sink.
    grid.set(11, 0, belt(Direction::East));
    for y in 0..4 {
        grid.set(12, y, belt(Direction::South));
    }

    // The main row: source, six belts east, assembler, output inserter, belt.
    grid.set(0, 4, port(Entity::Source, Item::IronPlate));
    for x in 1..=6 {
        grid.set(x, 4, belt(Direction::East));
    }
    grid.set(
        7,
        4,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::None,
            ..Default::default()
        },
    );
    grid.set(
        10,
        4,
        Cell {
            entity: Entity::Inserter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    grid.set(11, 4, belt(Direction::East));
    grid.set(12, 4, port(Entity::Sink, Item::IronGear));
    grid
}

/// Give the assembler the recipe the task calls for.
fn with_recipe(mut grid: Grid) -> Grid {
    let mut cell = grid.get(7, 4);
    cell.item = Item::IronGear;
    grid.set(7, 4, cell);
    grid
}

/// Swap the last belt of the feed line for an inserter that loads the machine.
///
/// The belt at (6,4) faces the assembler's wall and hands it nothing. An
/// inserter standing on the same tile picks up from the belt behind it and
/// swings the plate into the machine.
fn with_input_inserter(mut grid: Grid) -> Grid {
    grid.set(
        6,
        4,
        Cell {
            entity: Entity::Inserter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    grid
}

/// Delete the dead chain in the corner — the thing the issue suspected.
fn without_stray_chain(mut grid: Grid) -> Grid {
    grid.set(11, 0, Cell::default());
    for y in 0..4 {
        grid.set(12, y, Cell::default());
    }
    grid
}

fn report(label: &str, grid: &Grid) {
    let exports = blueprint::grid_to_blueprint(grid, "x").is_ok();
    println!(
        "  {label:<38} {:>7.3}/s  {:>5}  {:>5}  {:>6}",
        throughput::score(grid),
        if grid.is_consistent() { "yes" } else { "NO" },
        if item_reaches_sink(grid) { "yes" } else { "no" },
        if exports { "yes" } else { "NO" },
    );
}

fn main() {
    let base = example_one();
    println!("Example 1 from the issue, rebuilt from its blueprint string:\n");
    println!("{}", textual::render(&base));

    println!(
        "  {:<38} {:>9}  {:>5}  {:>5}  {:>6}",
        "", "items/s", "legal", "sink", "export"
    );
    report("as the model built it", &base);
    report(
        "without the two-belts-into-sink chain",
        &without_stray_chain(base.clone()),
    );
    report(
        "+ assembler given the gear recipe",
        &with_recipe(base.clone()),
    );
    report(
        "+ input inserter instead",
        &with_input_inserter(base.clone()),
    );
    report(
        "+ both repairs",
        &with_input_inserter(with_recipe(base.clone())),
    );
    println!("\nThe repaired layout:\n");
    println!(
        "{}",
        textual::render(&with_input_inserter(with_recipe(base)))
    );

    println!(
        "Read the second row first: deleting the chain the issue asked about changes\n\
         nothing, because a sink accepts from anything (`throughput::accepts_from`)\n\
         and that chain starts nowhere, so it carries nothing into it. Two belts\n\
         entering a sink is not a fault. It is untidy and it is free.\n\
         \n\
         The faults are the two rows after it, and neither repair helps alone:\n\
         \n\
         * The assembler has no recipe, and `sim::emits` is blunt about it: no\n\
           recipe, no craft. This used to be a *legal* cell — `is_consistent`\n\
           asked only that an item-bearing entity be allowed a tag, never that\n\
           it carry one — so a dead machine cost nothing on the `consistent`\n\
           metric and exported as an `assembling-machine-1` you would have to\n\
           click a recipe into by hand. An assembler now has to carry a tag that\n\
           names a real recipe, which drops the legal table from 57 rows to 45\n\
           and takes the dead machine away from the decoder as well as the\n\
           metric.\n\
         \n\
         * Six belts carry iron plate into the assembler's wall. Belts do not\n\
           load machines, in this simulator (`throughput.rs:267`) or in the game;\n\
           only an inserter does. The model built the output inserter correctly\n\
           and omitted the input one.\n\
         \n\
         The `NO` under `legal` and `export` is that first repair talking, and it\n\
         is the whole point: when the model drew this layout both columns read\n\
         `yes`, which is why it drew it. The same string the issue pasted is now\n\
         refused at the door (`blueprint.rs:103`) instead of exporting a factory\n\
         with an empty machine in it.\n\
         \n\
         The third row is worth its own look: `sink` says yes while the layout\n\
         delivers 0.000/s. That is not a contradiction, it is the gap between two\n\
         metrics. `item_reaches_sink` accepts from any pusher on purpose\n\
         (`sim.rs:57`) and does not know that belts cannot load a machine, so the\n\
         `functional` number in a training log scores layouts that deliver\n\
         nothing. Read `functional=0.912` with that in mind.\n\
         \n\
         Both repaired, the same layout runs. The model had the right plan --\n\
         source, line, machine, inserter, sink -- and lost on two details it is\n\
         never punished for at training time."
    );
}
