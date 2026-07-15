//! Procedural generation of known-correct factories ("lessons"), and blanking
//! them into (partial, solution) training pairs.
//!
//! This mirrors the reference project's data strategy
//! (`factorion_rs/src/factory_gen.rs`): each lesson is a distinct layout
//! pattern built by construction and verified functional (`sim::item_reaches_sink`),
//! then a subset of *removable* entities is blanked out. Source/sink/recipe
//! anchors are `protected` and never blanked, so the scaffold is always visible.
//!
//! For discrete diffusion the blanked cells become MASK tokens and the model
//! must inpaint them — the observed cells are the conditioning context.

use crate::sim::item_reaches_sink;
use crate::world::{Cell, Direction, Entity, Grid, Item, Misc};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

/// A lesson type — a family of factory layouts. Kept small and orthogonal so
/// every categorical channel (entity/direction/item/misc) is exercised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LessonKind {
    /// Source → belt path → sink. Exercises entity + direction.
    MoveOneItem,
    /// Same, but with random obstacles the belts must route around.
    MoveOneItemChaos,
    /// Source → inserter → assembler(recipe) → inserter → sink. Adds item.
    AssemblerLine,
    /// Source → belt → underground(down..up) → belt → sink across a wall.
    /// Exercises misc (underground tags).
    UndergroundCross,
    /// Up to [`BANK_LINES`] parallel assembler lines feeding one shared sink.
    /// Has **many** valid answers, and they are not equally good — see
    /// [`gen_assembler_bank`].
    AssemblerBank,
    /// Copper plate → cable → circuit ← iron plate. The only chain, the only
    /// two-input craft, and the only family whose best answer is *unbalanced* —
    /// see [`gen_circuit_line`].
    CircuitLine,
}

impl LessonKind {
    pub fn all() -> &'static [LessonKind] {
        &[
            LessonKind::MoveOneItem,
            LessonKind::MoveOneItemChaos,
            LessonKind::AssemblerLine,
            LessonKind::UndergroundCross,
            LessonKind::AssemblerBank,
            LessonKind::CircuitLine,
        ]
    }
    pub fn name(self) -> &'static str {
        match self {
            LessonKind::MoveOneItem => "MOVE_ONE_ITEM",
            LessonKind::MoveOneItemChaos => "MOVE_ONE_ITEM_CHAOS",
            LessonKind::AssemblerLine => "ASSEMBLER_LINE",
            LessonKind::UndergroundCross => "UNDERGROUND_CROSS",
            LessonKind::AssemblerBank => "ASSEMBLER_BANK",
            LessonKind::CircuitLine => "CIRCUIT_LINE",
        }
    }

    /// The smallest square grid this family fits on.
    ///
    /// Each generator already refuses a grid it cannot fit, so this is not what
    /// makes the curriculum correct — it is what keeps a caller from asking for
    /// a lesson that can never be built and burning [`generate`]'s whole retry
    /// budget discovering it. It lives here because the dimensions do; asked
    /// from `train.rs` it was a guess, and a wrong one (an assembler line is
    /// [`LINE_W`] columns wide, and was listed as needing five).
    pub fn min_size(self) -> usize {
        match self {
            LessonKind::MoveOneItem | LessonKind::MoveOneItemChaos => 3,
            LessonKind::AssemblerLine => LINE_W,
            LessonKind::UndergroundCross => 7,
            LessonKind::AssemblerBank => LINE_W.max(LINE_H * BANK_LINES),
            LessonKind::CircuitLine => CIRCUIT_W.max(CIRCUIT_H),
        }
    }

    /// Does this family admit more than one valid answer per task?
    ///
    /// Everywhere else the generator hands the model a task whose label is a
    /// *function* of the conditioning, which is why `experiments/task_space`
    /// measures zero ambiguous tasks and why `exact` and `functional` moved
    /// together for the whole 5,000-step run: there was only ever one answer, so
    /// getting it right and getting it working were the same event. A metric
    /// that ranks factories has nothing to do on data like that, and neither
    /// does a policy gradient.
    pub fn is_ambiguous(self) -> bool {
        matches!(self, LessonKind::AssemblerBank | LessonKind::CircuitLine)
    }
}

/// Parallel lines an [`LessonKind::AssemblerBank`] scaffold offers.
pub const BANK_LINES: usize = 3;

/// A generated, known-correct factory plus the bookkeeping needed to blank it.
#[derive(Debug, Clone)]
pub struct Sample {
    pub kind: LessonKind,
    /// The full correct factory (the diffusion target).
    pub solution: Grid,
    /// Cell indices that must never be blanked (source/sink/recipe anchors).
    pub protected: Vec<usize>,
    /// Cell indices holding removable entities (candidates for blanking).
    pub removable: Vec<usize>,
}

impl Sample {
    /// Blank `n` (or all removable if `n` is `None`) removable cells, producing a
    /// partial grid and an `observed` mask (`true` = given/conditioning cell,
    /// `false` = masked cell the model must generate).
    ///
    /// Returns `(partial_grid, observed_mask)`.
    pub fn blank(&self, n: Option<usize>, rng: &mut ChaCha8Rng) -> (Grid, Vec<bool>) {
        let mut removable = self.removable.clone();
        removable.shuffle(rng);
        let k = n.unwrap_or(removable.len()).min(removable.len());
        let blanked: &[usize] = &removable[..k];

        let mut partial = self.solution.clone();
        let mut observed = vec![true; self.solution.len()];
        for &i in blanked {
            partial.cells[i] = Cell::default();
            observed[i] = false;
        }
        (partial, observed)
    }

    /// Blank everything except the environment anchors: the source and sink
    /// tiles stay, every other cell is masked.
    ///
    /// [`Sample::blank`] only removes `removable` cells, so it always leaves the
    /// scaffold visible — the model is told where the assembler goes and which
    /// recipe it runs, and only fills a handful of gaps. That measures
    /// *inpainting*, not design. Here the model is instead given only "plates
    /// enter here, gears must arrive there" and has to decide what to build and
    /// where, which is the task we actually care about.
    ///
    /// Obstacles live on a separate, non-generative plane and stay visible;
    /// they are terrain, not something the model places. This mirrors the
    /// reference's honest `thput_eot` metric, which blanks the whole grid and
    /// rebuilds it from empty.
    pub fn blank_to_scaffold(&self) -> (Grid, Vec<bool>) {
        let mut partial = self.solution.clone();
        let observed: Vec<bool> = self
            .solution
            .cells
            .iter()
            .map(|cell| matches!(cell.entity, Entity::Source | Entity::Sink))
            .collect();
        for (cell, &keep) in partial.cells.iter_mut().zip(&observed) {
            if !keep {
                *cell = Cell::default();
            }
        }
        (partial, observed)
    }
}

/// Generate a functional factory for `kind` on a `size`×`size` grid, retrying
/// with fresh randomness until one validates. Deterministic in `seed`.
pub fn generate(kind: LessonKind, size: usize, seed: u64) -> Option<Sample> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let budget = (size * size * 40).max(500);
    for _ in 0..budget {
        let built = match kind {
            LessonKind::MoveOneItem => gen_move_one_item(size, &mut rng, false),
            LessonKind::MoveOneItemChaos => gen_move_one_item(size, &mut rng, true),
            LessonKind::AssemblerLine => gen_assembler_line(size, &mut rng),
            LessonKind::UndergroundCross => gen_underground_cross(size, &mut rng),
            LessonKind::AssemblerBank => gen_assembler_bank(size, &mut rng),
            LessonKind::CircuitLine => gen_circuit_line(size, &mut rng),
        };
        if let Some(sample) = built {
            debug_assert!(
                sample.solution.is_consistent(),
                "generated inconsistent factory"
            );
            debug_assert!(
                item_reaches_sink(&sample.solution),
                "generated non-functional factory"
            );
            return Some(sample);
        }
    }
    None
}

fn random_cell(size: usize, rng: &mut ChaCha8Rng) -> (usize, usize) {
    (rng.gen_range(0..size), rng.gen_range(0..size))
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

/// BFS shortest path over free cells (4-connected), avoiding obstacles and
/// occupied cells. `start` and `goal` are always passable endpoints. Returns the
/// path including both endpoints, or `None`.
fn bfs_path(
    grid: &Grid,
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    let passable = |x: usize, y: usize| -> bool {
        if (x, y) == start || (x, y) == goal {
            return true;
        }
        !grid.is_obstacle(x, y) && grid.get(x, y).is_empty()
    };
    let mut prev = vec![usize::MAX; grid.len()];
    let mut seen = vec![false; grid.len()];
    let mut q = VecDeque::new();
    seen[grid.idx(start.0, start.1)] = true;
    q.push_back(start);
    while let Some((x, y)) = q.pop_front() {
        if (x, y) == goal {
            // reconstruct
            let mut path = vec![goal];
            let mut cur = grid.idx(x, y);
            while cur != grid.idx(start.0, start.1) {
                let p = prev[cur];
                path.push((p % grid.width, p / grid.width));
                cur = p;
            }
            path.reverse();
            return Some(path);
        }
        for d in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            let (dx, dy) = d.delta();
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if grid.in_bounds(nx, ny) {
                let (nx, ny) = (nx as usize, ny as usize);
                let i = grid.idx(nx, ny);
                if !seen[i] && passable(nx, ny) {
                    seen[i] = true;
                    prev[i] = grid.idx(x, y);
                    q.push_back((nx, ny));
                }
            }
        }
    }
    None
}

/// Direction pointing from `a` to orthogonally-adjacent `b`.
fn dir_between(a: (usize, usize), b: (usize, usize)) -> Direction {
    let dx = b.0 as i32 - a.0 as i32;
    let dy = b.1 as i32 - a.1 as i32;
    match (dx, dy) {
        (0, -1) => Direction::North,
        (1, 0) => Direction::East,
        (0, 1) => Direction::South,
        (-1, 0) => Direction::West,
        _ => Direction::None,
    }
}

fn gen_move_one_item(size: usize, rng: &mut ChaCha8Rng, chaos: bool) -> Option<Sample> {
    if size < 3 {
        return None;
    }
    let mut grid = Grid::new(size, size);

    if chaos {
        // Sprinkle a few obstacles (~10% of cells).
        let n_obstacles = (size * size / 10).max(1);
        for _ in 0..n_obstacles {
            let (x, y) = random_cell(size, rng);
            grid.set_obstacle(x, y, true);
        }
    }

    let source = random_cell(size, rng);
    if grid.is_obstacle(source.0, source.1) {
        return None;
    }
    let sink = random_cell(size, rng);
    if grid.is_obstacle(sink.0, sink.1) || manhattan(source, sink) < 2 {
        return None;
    }

    let item = *[Item::IronPlate, Item::CopperPlate, Item::GreenCircuit]
        .choose(rng)
        .unwrap();
    grid.set(
        source.0,
        source.1,
        Cell {
            entity: Entity::Source,
            item,
            ..Default::default()
        },
    );
    grid.set(
        sink.0,
        sink.1,
        Cell {
            entity: Entity::Sink,
            item,
            ..Default::default()
        },
    );

    let path = bfs_path(&grid, source, sink)?;
    // Interior cells (exclude the source and sink endpoints) become belts.
    let interior = &path[1..path.len() - 1];
    if interior.is_empty() {
        return None; // adjacent source/sink: no belts to learn
    }
    let mut removable = Vec::new();
    for (i, &pos) in interior.iter().enumerate() {
        let next = path[i + 2]; // cell after this interior cell in the path
        let dir = dir_between(pos, next);
        grid.set(pos.0, pos.1, Cell::belt(dir));
        removable.push(grid.idx(pos.0, pos.1));
    }

    if !item_reaches_sink(&grid) {
        return None;
    }
    let protected = vec![grid.idx(source.0, source.1), grid.idx(sink.0, sink.1)];
    Some(Sample {
        kind: if chaos {
            LessonKind::MoveOneItemChaos
        } else {
            LessonKind::MoveOneItem
        },
        solution: grid,
        protected,
        removable,
    })
}

/// Columns one crafting line occupies: source, inserter, the machine's three,
/// inserter, sink.
const LINE_W: usize = 7;

/// Rows one crafting line occupies — the machine's three. The line itself runs
/// along the middle one.
const LINE_H: usize = 3;

/// A crafting line, with the machine covering the nine tiles a real
/// `assembling-machine-1` covers:
///
/// ```text
///   . . A A A . .
///   S i A A A i K
///   . . A A A . .
/// ```
///
/// This lesson used to be `S i a i K` on a single row, with the assembler
/// occupying one cell "simplified from the reference's 3×3 to keep the first
/// model tractable". The simplification was not free and not local: `blueprint.rs`
/// has always anchored a *real* 3×3 `assembling-machine-1` at that cell, so every
/// blueprint this lesson ever exported placed the machine on top of its own two
/// inserters and Factorio rejected the import outright
/// (`experiments/overlap_check.rs` reproduces it). The model was being taught a
/// shape that cannot be built.
///
/// Only the anchor at (x0+2, y0) stores the machine; the other eight tiles stay
/// `Empty` and are reached through [`Grid::anchor_at`]. See `world.rs` for why
/// the shadow is implied rather than stamped.
fn gen_assembler_line(size: usize, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if size < LINE_W || size < LINE_H {
        return None;
    }
    let y0 = rng.gen_range(0..=(size - LINE_H));
    let x0 = rng.gen_range(0..=(size - LINE_W));
    // One source feeds one machine, so this lesson can only teach the recipes
    // that need one thing. The two-input recipes are [`LessonKind::CircuitLine`].
    let recipe = *Item::single_input_craftable().choose(rng).unwrap();
    let input_item = recipe.ingredients()[0].item;

    // The machine's middle row: the one the line runs along.
    let y = y0 + 1;
    let mut grid = Grid::new(size, size);
    grid.set(
        x0,
        y,
        Cell {
            entity: Entity::Source,
            item: input_item,
            ..Default::default()
        },
    );
    grid.set(
        x0 + 1,
        y,
        Cell {
            entity: Entity::Inserter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    grid.set(
        x0 + 2,
        y0,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: recipe,
            misc: Misc::None,
        },
    );
    grid.set(
        x0 + 5,
        y,
        Cell {
            entity: Entity::Inserter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    grid.set(
        x0 + 6,
        y,
        Cell {
            entity: Entity::Sink,
            item: recipe,
            ..Default::default()
        },
    );

    if !item_reaches_sink(&grid) {
        return None;
    }
    // Assembler (recipe) is protected; the two inserters are removable.
    let protected = vec![grid.idx(x0, y), grid.idx(x0 + 2, y0), grid.idx(x0 + 6, y)];
    let removable = vec![grid.idx(x0 + 1, y), grid.idx(x0 + 5, y)];
    Some(Sample {
        kind: LessonKind::AssemblerLine,
        solution: grid,
        protected,
        removable,
    })
}

/// A bank of parallel assembler lines feeding one shared sink, each machine
/// three tiles on a side and so each line three rows tall:
///
/// ```text
///   . . A A A . .
///   S i A A A i b(South)
///   . . A A A . b(South)
///   . . A A A . b(South)
///   S i A A A i b(South)
///   . . A A A . b(South)
///   . . A A A . b(South)
///   S i A A A i K
///   . . A A A . .
/// ```
///
/// The scaffold — what stays observed — is only the three sources and the sink.
/// **How many of the three lines get built is up to the answer**, and the
/// generator picks that number at random, so the same task appears with a
/// one-line answer, a two-line answer and a three-line answer. This is the one
/// thing every other family lacks and the reason none of the machinery
/// downstream had anything to do:
///
/// * The answers are not equally good. Each line adds its machine's output to
///   the shared sink, so three lines deliver three times what one does. That is
///   a *gradient over working factories* — the thing `functional` cannot see and
///   [`crate::throughput`] can.
/// * `exact` cannot reach 1.0 here, and should not: matching one arbitrary draw
///   out of three is not a skill. `functional` still can, which is precisely why
///   the two metrics finally say different things.
/// * A model that learns the distribution can be asked for eight draws and
///   handed the best one ([`crate::best_of_n`]) — which will usually be a
///   three-line factory even when the taught answer had one line. That is a
///   factory nobody put in the data, built from nothing but "plates arrive
///   here, gears are wanted there".
///
/// A source stands in for an unlimited supply, so all three lines can run flat
/// out; the ceiling is the input inserter's 0.86 items/s per line, exactly as it
/// would be in Factorio.
fn gen_assembler_bank(size: usize, rng: &mut ChaCha8Rng) -> Option<Sample> {
    // Lines stack back to back: three rows each, and no gap between them, so a
    // bank of three is nine rows tall. The columns are a line's own, plus one
    // for the belt that merges every line into the shared sink.
    let height = LINE_H * BANK_LINES;
    let width = LINE_W;
    if size < width || size < height {
        return None;
    }
    let y0 = rng.gen_range(0..=(size - height));
    let x0 = rng.gen_range(0..=(size - width));
    // Every line here is fed by one source, so — as in [`gen_assembler_line`] —
    // only the single-input recipes fit.
    let recipe = *Item::single_input_craftable().choose(rng).unwrap();
    let input_item = recipe.ingredients()[0].item;

    // Line `j` is anchored at `machine(j)` and runs along `row(j)`, its middle.
    let machine = |j: usize| (x0 + 2, y0 + LINE_H * j);
    let row = |j: usize| y0 + LINE_H * j + 1;
    let column = x0 + width - 1;
    let sink = (column, row(BANK_LINES - 1));

    let mut grid = Grid::new(size, size);
    let mut protected = Vec::new();
    for j in 0..BANK_LINES {
        grid.set(
            x0,
            row(j),
            Cell {
                entity: Entity::Source,
                item: input_item,
                ..Default::default()
            },
        );
        protected.push(grid.idx(x0, row(j)));
    }
    grid.set(
        sink.0,
        sink.1,
        Cell {
            entity: Entity::Sink,
            item: recipe,
            ..Default::default()
        },
    );
    protected.push(grid.idx(sink.0, sink.1));

    // `removable` is the region the answer may write to, not the region this
    // answer happened to fill. The distinction is the whole family: `blank`
    // observes every cell it does not blank, so listing only the built cells
    // would leave an unbuilt line *given* as empty — the conditioning would
    // spell out the line count and the ambiguity would evaporate. Masking the
    // whole bank instead asks the model the question we mean to ask: how many
    // lines belong here?
    let mut removable = Vec::new();
    for y in y0..y0 + height {
        for x in x0..x0 + width {
            let i = grid.idx(x, y);
            if !protected.contains(&i) {
                removable.push(i);
            }
        }
    }

    // The choice that makes this family ambiguous. Lines are built upward from
    // the sink's own row so the belt column is always unbroken; leaving the
    // count to the answer rather than to the scaffold is the entire point.
    let lines = rng.gen_range(1..=BANK_LINES);
    let first = BANK_LINES - lines;
    for j in first..BANK_LINES {
        grid.set(
            x0 + 1,
            row(j),
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        let (mx, my) = machine(j);
        grid.set(
            mx,
            my,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                item: recipe,
                misc: Misc::None,
            },
        );
        grid.set(
            x0 + 5,
            row(j),
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
    }
    // One belt column carries every line's output down into the shared sink. It
    // starts at the topmost line that was actually built: the bottom line hands
    // off to the sink directly and needs no belt at all.
    for y in row(first)..sink.1 {
        grid.set(column, y, Cell::belt(Direction::South));
    }

    if !item_reaches_sink(&grid) {
        return None;
    }
    Some(Sample {
        kind: LessonKind::AssemblerBank,
        solution: grid,
        protected,
        removable,
    })
}

/// Columns a [`LessonKind::CircuitLine`] occupies: copper source, its inserter,
/// the cable machine's three, the shared inserter column, the circuit machine's
/// three, the output inserter, the sink.
const CIRCUIT_W: usize = 11;

/// Rows it occupies: the iron source and its inserter stacked above the two
/// machines' three.
const CIRCUIT_H: usize = 5;

/// Cable inserters the shared column offers. The machines' faces meet there, so
/// there are exactly as many slots as a machine is tall.
const CABLE_FEEDS: usize = 3;

/// An electronic circuit, built from iron and copper the way Factorio builds it:
///
/// ```text
///   .  .  .  .  .  .  Fe .  .  .  .
///   .  .  .  .  .  .  i  .  .  .  .     <- south, into the circuit machine
///   .  .  C  C  C  i  A  A  A  .  .
///   Cu i  C  C  C  i  A  A  A  i  K
///   .  .  C  C  C  i  A  A  A  .  .
/// ```
///
/// `C` crafts copper cable from copper plate; `A` crafts the circuit from iron
/// plate **and** that cable ([wiki](https://wiki.factorio.com/Electronic_circuit)).
/// Two things here exist nowhere else in the curriculum:
///
/// * **A chain.** Two machines, two different recipes, and the second one's
///   input is the first one's output. Every other lesson crafts in one step from
///   a raw plate, so "what recipe goes here" was always read straight off the
///   sink's own tag. Here the cable machine's recipe is not written down
///   anywhere — it has to be derived from what the circuit needs.
/// * **A join.** The machine runs only when *both* inputs arrive, so neither
///   feed can be sacrificed for the other. A single-input line is a path; this
///   is two paths that must both land, which is why [`crate::sim`] had to stop
///   walking one item at a time and start reasoning about what has arrived.
///
/// The geometry is not a coincidence: the cable machine's east face and the
/// circuit machine's west face are *the same column*, so an inserter placed
/// there both unloads one machine and loads the other, and [`CABLE_FEEDS`] of
/// them fit.
///
/// **How many is up to the answer**, and that is the second ambiguous family.
/// The count matters because the recipe is unbalanced — one craft eats 1 plate
/// and 3 cable — so a layout that feeds both inputs the same way starves on
/// cable while iron piles up. One feed carries 0.86 cable/s and the machine
/// wants 3× its iron rate, so it crafts at a third of what the iron alone would
/// support; a second feed doubles the factory's output for one entity. A third
/// adds nothing, because by then the *copper* inserter into the cable machine is
/// the binding constraint — which is a fact about the layout that only a graded
/// score can state, and `functional` calls all three answers perfect.
fn gen_circuit_line(size: usize, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if size < CIRCUIT_W || size < CIRCUIT_H {
        return None;
    }
    let x0 = rng.gen_range(0..=(size - CIRCUIT_W));
    let y0 = rng.gen_range(0..=(size - CIRCUIT_H));

    // The row the line runs along: the machines' middle.
    let y = y0 + 3;
    let cable = (x0 + 2, y0 + 2);
    let circuit = (x0 + 6, y0 + 2);
    let iron_source = (x0 + 6, y0);
    let copper_source = (x0, y);
    let sink = (x0 + CIRCUIT_W - 1, y);

    let mut grid = Grid::new(size, size);
    grid.set(
        copper_source.0,
        copper_source.1,
        Cell {
            entity: Entity::Source,
            item: Item::CopperPlate,
            ..Default::default()
        },
    );
    grid.set(
        iron_source.0,
        iron_source.1,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    grid.set(
        sink.0,
        sink.1,
        Cell {
            entity: Entity::Sink,
            item: Item::GreenCircuit,
            ..Default::default()
        },
    );
    let protected = vec![
        grid.idx(copper_source.0, copper_source.1),
        grid.idx(iron_source.0, iron_source.1),
        grid.idx(sink.0, sink.1),
    ];

    // As in `gen_assembler_bank`: everything the answer *may* write to, not
    // everything this answer happened to write. Listing only the built cells
    // would hand the model the cable count as conditioning and the family would
    // stop being ambiguous.
    let mut removable = Vec::new();
    for gy in y0..y0 + CIRCUIT_H {
        for gx in x0..x0 + CIRCUIT_W {
            let i = grid.idx(gx, gy);
            if !protected.contains(&i) {
                removable.push(i);
            }
        }
    }

    // Copper plate in.
    grid.set(x0 + 1, y, inserter(Direction::East));
    grid.set(
        cable.0,
        cable.1,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::CopperCable,
            misc: Misc::None,
        },
    );
    // Iron plate down into the circuit machine's north face.
    grid.set(x0 + 6, y0 + 1, inserter(Direction::South));
    grid.set(
        circuit.0,
        circuit.1,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::GreenCircuit,
            misc: Misc::None,
        },
    );
    // The choice that makes this family ambiguous.
    let feeds = rng.gen_range(1..=CABLE_FEEDS);
    for j in 0..feeds {
        grid.set(x0 + 5, y0 + 2 + j, inserter(Direction::East));
    }
    // Circuits out.
    grid.set(x0 + 9, y, inserter(Direction::East));

    if !item_reaches_sink(&grid) {
        return None;
    }
    Some(Sample {
        kind: LessonKind::CircuitLine,
        solution: grid,
        protected,
        removable,
    })
}

fn inserter(direction: Direction) -> Cell {
    Cell {
        entity: Entity::Inserter,
        direction,
        ..Default::default()
    }
}

fn gen_underground_cross(size: usize, rng: &mut ChaCha8Rng) -> Option<Sample> {
    // Horizontal line with an obstacle wall the belts tunnel under:
    // S b d # u b K   (wall at the '#').
    if size < 7 {
        return None;
    }
    let y = rng.gen_range(0..size);
    let x0 = rng.gen_range(0..=(size - 7));
    let item = *[Item::IronPlate, Item::CopperPlate].choose(rng).unwrap();

    let mut grid = Grid::new(size, size);
    let wall = x0 + 3;
    grid.set_obstacle(wall, y, true);

    grid.set(
        x0,
        y,
        Cell {
            entity: Entity::Source,
            item,
            ..Default::default()
        },
    );
    grid.set(x0 + 1, y, Cell::belt(Direction::East));
    grid.set(
        x0 + 2,
        y,
        Cell {
            entity: Entity::UndergroundBelt,
            direction: Direction::East,
            misc: Misc::UndergroundDown,
            ..Default::default()
        },
    );
    grid.set(
        x0 + 4,
        y,
        Cell {
            entity: Entity::UndergroundBelt,
            direction: Direction::East,
            misc: Misc::UndergroundUp,
            ..Default::default()
        },
    );
    grid.set(x0 + 5, y, Cell::belt(Direction::East));
    grid.set(
        x0 + 6,
        y,
        Cell {
            entity: Entity::Sink,
            item,
            ..Default::default()
        },
    );

    if !item_reaches_sink(&grid) {
        return None;
    }
    let protected = vec![grid.idx(x0, y), grid.idx(x0 + 6, y)];
    // The two belts and the two underground endpoints are removable.
    let removable = vec![
        grid.idx(x0 + 1, y),
        grid.idx(x0 + 2, y),
        grid.idx(x0 + 4, y),
        grid.idx(x0 + 5, y),
    ];
    Some(Sample {
        kind: LessonKind::UndergroundCross,
        solution: grid,
        protected,
        removable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::throughput;
    use std::collections::{HashMap, HashSet};

    /// What the model conditions on: the cells left observed, plus terrain.
    fn conditioning(sample: &Sample, observed: &[bool]) -> String {
        let mut key = String::new();
        for (i, &obs) in observed.iter().enumerate() {
            if obs {
                let c = sample.solution.cells[i];
                key.push_str(&format!(
                    "{i}:{}:{}:{}:{};",
                    c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
                ));
            }
        }
        for (i, &blocked) in sample.solution.obstacle.iter().enumerate() {
            if blocked {
                key.push_str(&format!("#{i};"));
            }
        }
        key
    }

    /// The answer it must produce on the masked cells.
    fn answer(sample: &Sample, observed: &[bool]) -> String {
        let mut key = String::new();
        for (i, &obs) in observed.iter().enumerate() {
            if !obs {
                let c = sample.solution.cells[i];
                key.push_str(&format!(
                    "{i}:{}:{}:{}:{};",
                    c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
                ));
            }
        }
        key
    }

    /// The whole reason this family exists. `experiments/task_space` measures
    /// zero ambiguous tasks across the original four: every task's label is a
    /// function of its conditioning, so `exact` is always reachable, `functional`
    /// tracks it exactly, and a ranking metric has nothing to rank. Here one task
    /// has several valid answers, which is the precondition for everything
    /// downstream — Best-of-N needs a choice to make, and RL needs a reward that
    /// is not already saturated.
    ///
    /// Checked under *both* blankings, and the inpainting one is the load-bearing
    /// case. [`Sample::blank`] observes every cell it does not blank, so a family
    /// that lists only its built cells as `removable` hands the model an unbuilt
    /// line as a given empty tile — the conditioning states the line count and
    /// the ambiguity is gone. That bug passes a `blank_to_scaffold`-only test
    /// (which observes nothing but the anchors) and shows up in training, where
    /// `blank` is what runs.
    #[test]
    fn the_assembler_bank_gives_one_task_several_valid_answers() {
        for (blanking, observe) in [
            (
                "scaffold",
                &(|s: &Sample| s.blank_to_scaffold().1) as &dyn Fn(&Sample) -> Vec<bool>,
            ),
            ("inpaint", &|s: &Sample| {
                s.blank(None, &mut ChaCha8Rng::seed_from_u64(0)).1
            }),
        ] {
            let mut answers: HashMap<String, HashSet<String>> = HashMap::new();
            for seed in 0..2_000u64 {
                let Some(sample) = generate(LessonKind::AssemblerBank, 11, seed) else {
                    continue;
                };
                let observed = observe(&sample);
                answers
                    .entry(conditioning(&sample, &observed))
                    .or_default()
                    .insert(answer(&sample, &observed));
            }

            let ambiguous = answers.values().filter(|a| a.len() > 1).count();
            assert!(
                ambiguous > 0,
                "{blanking}: every task still has exactly one answer -- the family is pointless"
            );
            // Not a fluke on one task: the family is ambiguous by construction, so
            // essentially every task it produces should be.
            assert!(
                ambiguous * 4 > answers.len() * 3,
                "{blanking}: only {ambiguous} of {} tasks admit more than one answer",
                answers.len()
            );
        }
        assert!(LessonKind::AssemblerBank.is_ambiguous());
        // Every *other* family hands the model a task whose label is a function
        // of the conditioning. `is_ambiguous` has to keep saying so, because
        // `experiments/task_space` and every claim in `docs/RL_ANALYSIS.md` rest
        // on it.
        assert!(LessonKind::all()
            .iter()
            .filter(|k| k.is_ambiguous())
            .all(|k| matches!(k, LessonKind::AssemblerBank | LessonKind::CircuitLine)));
    }

    /// Ambiguity alone is not enough: if every answer delivered the same rate,
    /// ranking them would still be a coin flip. The answers have to be *unequal*,
    /// and by a real margin — that margin is the gradient Best-of-N climbs and
    /// the one `beat_original` reports.
    #[test]
    fn the_bank_answers_are_not_equally_good() {
        let mut by_rate: HashMap<String, Vec<f64>> = HashMap::new();
        for seed in 0..2_000u64 {
            let Some(sample) = generate(LessonKind::AssemblerBank, 11, seed) else {
                continue;
            };
            let (_, observed) = sample.blank_to_scaffold();
            by_rate
                .entry(conditioning(&sample, &observed))
                .or_default()
                .push(throughput::score(&sample.solution));
        }

        let spread = by_rate
            .values()
            .filter(|rates| {
                let (lo, hi) = (
                    rates.iter().copied().fold(f64::INFINITY, f64::min),
                    rates.iter().copied().fold(0.0, f64::max),
                );
                // Each extra line adds a whole machine's output, so the best
                // answer to a task should deliver multiples of the worst.
                hi > lo * 1.9
            })
            .count();
        assert!(
            spread * 2 > by_rate.len(),
            "only {spread} of {} tasks have answers that differ in rate",
            by_rate.len()
        );
    }

    /// Lines are supposed to add up rather than fight over the sink. If the belt
    /// column ever failed to merge them, ambiguity would still be there but the
    /// rate ranking would be noise, and every conclusion drawn from it wrong.
    #[test]
    fn each_extra_line_in_the_bank_adds_its_own_output() {
        // Rate as a function of how many lines the answer built. Recipes differ
        // in rate, so group by recipe and compare like with like.
        let mut by_recipe: HashMap<(u8, usize), f64> = HashMap::new();
        for seed in 0..2_000u64 {
            let Some(sample) = generate(LessonKind::AssemblerBank, 11, seed) else {
                continue;
            };
            let lines = sample
                .solution
                .cells
                .iter()
                .filter(|c| c.entity == Entity::Assembler)
                .count();
            let recipe = sample
                .solution
                .cells
                .iter()
                .find(|c| c.entity == Entity::Sink)
                .expect("bank always has a sink")
                .item as u8;
            let rate = throughput::score(&sample.solution);
            let previous = by_recipe.insert((recipe, lines), rate);
            if let Some(previous) = previous {
                assert!(
                    (previous - rate).abs() < 1e-9,
                    "{lines} lines of recipe {recipe} delivered {previous} and {rate}"
                );
            }
        }

        // A bank line is one source into one machine, so it can only run the
        // single-input recipes — the two-input ones are `CIRCUIT_LINE`'s job.
        for recipe in Item::single_input_craftable() {
            let one = by_recipe[&(recipe as u8, 1)];
            assert!(one > 0.0, "a one-line bank delivers nothing");
            for lines in 2..=BANK_LINES {
                let rate = by_recipe[&(recipe as u8, lines)];
                assert!(
                    (rate - one * lines as f64).abs() < 1e-9,
                    "{lines} lines delivered {rate}, not {} ({lines}x the one-line {one})",
                    one * lines as f64
                );
            }
        }
    }

    #[test]
    fn all_lessons_generate_functional_factories() {
        for &kind in LessonKind::all() {
            let mut ok = 0;
            for seed in 0..40u64 {
                if let Some(s) = generate(kind, 11, seed) {
                    assert!(s.solution.is_consistent(), "{}: inconsistent", kind.name());
                    assert!(
                        item_reaches_sink(&s.solution),
                        "{}: not functional",
                        kind.name()
                    );
                    assert!(
                        !s.removable.is_empty(),
                        "{}: nothing removable",
                        kind.name()
                    );
                    ok += 1;
                }
            }
            assert!(ok > 0, "lesson {} never generated", kind.name());
        }
    }

    #[test]
    fn blanking_masks_removable_cells() {
        let s = generate(LessonKind::MoveOneItem, 11, 7).expect("gen");
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (partial, observed) = s.blank(None, &mut rng);
        // Every removable cell is now empty and unobserved.
        for &i in &s.removable {
            assert!(partial.cells[i].is_empty());
            assert!(!observed[i]);
        }
        // Protected cells are untouched and observed.
        for &i in &s.protected {
            assert!(observed[i]);
            assert_eq!(partial.cells[i], s.solution.cells[i]);
        }
    }

    #[test]
    fn blanking_to_scaffold_leaves_only_the_source_and_sink() {
        for &kind in LessonKind::all() {
            let Some(s) = generate(kind, 11, 7) else {
                continue;
            };
            let (partial, observed) = s.blank_to_scaffold();
            for (i, cell) in s.solution.cells.iter().enumerate() {
                if matches!(cell.entity, Entity::Source | Entity::Sink) {
                    assert!(observed[i], "{}: anchor was blanked", kind.name());
                    assert_eq!(partial.cells[i], *cell);
                } else {
                    assert!(!observed[i], "{}: cell {i} stayed visible", kind.name());
                    assert!(partial.cells[i].is_empty());
                }
            }
            // Obstacles are terrain on a separate plane, not something the model
            // places, so they survive blanking.
            assert_eq!(partial.obstacle, s.solution.obstacle);
        }
    }

    /// The gap between the two modes is the whole point: `blank` leaves the
    /// scaffold up and asks for a handful of cells, so a model can score well on
    /// it without ever having decided what to build.
    #[test]
    fn blanking_to_scaffold_asks_for_far_more_than_inpainting() {
        let s = generate(LessonKind::AssemblerLine, 11, 7).expect("gen");
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (_, inpaint) = s.blank(None, &mut rng);
        let (_, scratch) = s.blank_to_scaffold();

        let masked = |observed: &[bool]| observed.iter().filter(|&&o| !o).count();
        assert!(
            masked(&scratch) > 10 * masked(&inpaint),
            "from scratch masked {} cells, inpainting masked {}",
            masked(&scratch),
            masked(&inpaint)
        );
        // The assembler and its recipe tag are gone too, so the only clue to what
        // to craft is the item the sink accepts.
        assert!(s
            .solution
            .cells
            .iter()
            .zip(&scratch)
            .any(|(cell, &obs)| cell.entity == Entity::Assembler && !obs));
    }

    /// How many cable inserters a circuit line's answer built.
    fn cable_feeds(sample: &Sample) -> usize {
        sample
            .solution
            .cells
            .iter()
            .enumerate()
            .filter(|(i, c)| {
                c.entity == Entity::Inserter && {
                    // The shared column: the one whose inserters unload the cable
                    // machine. An inserter is there iff it picks up from an
                    // assembler and drops into one.
                    let (x, y) = (i % sample.solution.width, i / sample.solution.width);
                    let (dx, dy) = c.direction.delta();
                    let from = sample
                        .solution
                        .anchor_at((x as i32 - dx) as usize, (y as i32 - dy) as usize);
                    let to = sample
                        .solution
                        .anchor_at((x as i32 + dx) as usize, (y as i32 + dy) as usize);
                    let machine = |a: Option<(usize, usize)>| {
                        a.is_some_and(|(ax, ay)| {
                            sample.solution.get(ax, ay).entity == Entity::Assembler
                        })
                    };
                    machine(from) && machine(to)
                }
            })
            .count()
    }

    /// The circuit line is the only lesson that crafts from two different inputs,
    /// and the assertion that matters is that the generator never emits one the
    /// simulator would call broken — a machine fed iron but no cable crafts
    /// nothing, and `generate`'s own `debug_assert` would have caught it, but
    /// only in debug. CI runs `--release`.
    #[test]
    fn every_circuit_line_actually_delivers_circuits() {
        let mut built = 0;
        for seed in 0..200u64 {
            let Some(sample) = generate(LessonKind::CircuitLine, 11, seed) else {
                continue;
            };
            built += 1;
            assert!(
                sample.solution.is_consistent(),
                "seed {seed} is unbuildable"
            );
            assert!(
                sample.solution.footprints_are_legal(),
                "seed {seed} overlaps its own machines"
            );
            assert!(
                item_reaches_sink(&sample.solution),
                "seed {seed} does not deliver"
            );
            assert!(
                throughput::score(&sample.solution) > 0.0,
                "seed {seed} routes but delivers nothing"
            );
        }
        assert_eq!(built, 200, "the generator failed on some seeds");
    }

    /// The claim [`gen_circuit_line`]'s doc makes, checked rather than asserted
    /// in prose: one cable feed starves the machine at a third of what its iron
    /// supports, a second doubles the factory, and a third adds nothing because
    /// the copper inserter upstream has become the constraint.
    ///
    /// This is the shape no single-input lesson can produce. A bank line's rate
    /// is linear in the entity count — twice the lines, twice the output — so
    /// "build more" is the whole lesson. Here the *same* entity is worth 2x, 1x
    /// and nothing depending on where the bottleneck already is, and `functional`
    /// scores all three answers identically.
    #[test]
    fn the_second_cable_feed_doubles_the_circuit_line_and_the_third_does_not() {
        let mut by_feeds: HashMap<usize, f64> = HashMap::new();
        for seed in 0..200u64 {
            let Some(sample) = generate(LessonKind::CircuitLine, 11, seed) else {
                continue;
            };
            let rate = throughput::score(&sample.solution);
            let feeds = cable_feeds(&sample);
            if let Some(previous) = by_feeds.insert(feeds, rate) {
                assert!(
                    (previous - rate).abs() < 1e-9,
                    "{feeds} feeds delivered {previous} and {rate}"
                );
            }
            // However few circuits it makes, it makes them.
            assert!(item_reaches_sink(&sample.solution));
        }
        assert_eq!(
            by_feeds.len(),
            CABLE_FEEDS,
            "not every feed count was drawn"
        );

        let (one, two, three) = (by_feeds[&1], by_feeds[&2], by_feeds[&3]);
        assert!(
            (two - one * 2.0).abs() < 1e-9,
            "a second feed took {one} to {two}, not to {}",
            one * 2.0
        );
        assert!(
            (three - two).abs() < 1e-9,
            "a third feed took {two} to {three}; the copper inserter should have capped it"
        );
    }

    /// The join, stated as a fact about the simulator rather than the generator:
    /// take a working line and delete the iron feed, and it stops making
    /// circuits even though the cable still arrives and the machine is still
    /// there.
    ///
    /// The old single-item walk could not have failed this test, because it
    /// could not have passed the one above it: it asked "am I carrying the
    /// ingredient?" and a two-ingredient recipe has no answer to that.
    #[test]
    fn a_circuit_machine_starved_of_either_input_crafts_nothing() {
        let sample = generate(LessonKind::CircuitLine, 11, 0).expect("gen");
        assert!(item_reaches_sink(&sample.solution));

        for starve in [Item::IronPlate, Item::CopperPlate] {
            let mut grid = sample.solution.clone();
            let source = grid
                .cells
                .iter()
                .position(|c| c.entity == Entity::Source && c.item == starve)
                .expect("both sources exist");
            grid.cells[source] = Cell::default();
            assert!(
                !item_reaches_sink(&grid),
                "cut the {starve:?} feed and it still delivered circuits"
            );
        }
    }
}
