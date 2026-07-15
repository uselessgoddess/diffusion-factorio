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
    /// The only family with **many** valid answers, and they are not equally
    /// good — see [`gen_assembler_bank`].
    AssemblerBank,
}

impl LessonKind {
    pub fn all() -> &'static [LessonKind] {
        &[
            LessonKind::MoveOneItem,
            LessonKind::MoveOneItemChaos,
            LessonKind::AssemblerLine,
            LessonKind::UndergroundCross,
            LessonKind::AssemblerBank,
        ]
    }
    pub fn name(self) -> &'static str {
        match self {
            LessonKind::MoveOneItem => "MOVE_ONE_ITEM",
            LessonKind::MoveOneItemChaos => "MOVE_ONE_ITEM_CHAOS",
            LessonKind::AssemblerLine => "ASSEMBLER_LINE",
            LessonKind::UndergroundCross => "UNDERGROUND_CROSS",
            LessonKind::AssemblerBank => "ASSEMBLER_BANK",
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
        matches!(self, LessonKind::AssemblerBank)
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

fn gen_assembler_line(size: usize, rng: &mut ChaCha8Rng) -> Option<Sample> {
    // Horizontal line: S i a i K  (needs width >= 5). Assembler is 1×1 here
    // (simplified from the reference's 3×3) to keep the first model tractable.
    if size < 5 {
        return None;
    }
    let y = rng.gen_range(0..size);
    let x0 = rng.gen_range(0..=(size - 5));
    let recipe = *Item::craftable().choose(rng).unwrap();
    let input_item = recipe
        .ingredient()
        .expect("every craftable recipe has an ingredient");

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
        y,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: recipe,
            misc: Misc::None,
        },
    );
    grid.set(
        x0 + 3,
        y,
        Cell {
            entity: Entity::Inserter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    grid.set(
        x0 + 4,
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
    let protected = vec![grid.idx(x0, y), grid.idx(x0 + 2, y), grid.idx(x0 + 4, y)];
    let removable = vec![grid.idx(x0 + 1, y), grid.idx(x0 + 3, y)];
    Some(Sample {
        kind: LessonKind::AssemblerLine,
        solution: grid,
        protected,
        removable,
    })
}

/// A bank of parallel assembler lines feeding one shared sink:
///
/// ```text
///   S i a i b(South)
///   S i a i b(South)
///   S i a i K
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
    // Width: source, inserter, assembler, inserter, sink/belt column.
    if size < 5 || size < BANK_LINES {
        return None;
    }
    let y0 = rng.gen_range(0..=(size - BANK_LINES));
    let x0 = rng.gen_range(0..=(size - 5));
    let recipe = *Item::craftable().choose(rng).unwrap();
    let input_item = recipe
        .ingredient()
        .expect("every craftable recipe has an ingredient");

    let mut grid = Grid::new(size, size);
    let sink = (x0 + 4, y0 + BANK_LINES - 1);
    let mut protected = Vec::new();
    for j in 0..BANK_LINES {
        grid.set(
            x0,
            y0 + j,
            Cell {
                entity: Entity::Source,
                item: input_item,
                ..Default::default()
            },
        );
        protected.push(grid.idx(x0, y0 + j));
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

    // The choice that makes this family ambiguous. Lines are built upward from
    // the sink's own row so the belt column is always unbroken; leaving the
    // count to the answer rather than to the scaffold is the entire point.
    let lines = rng.gen_range(1..=BANK_LINES);
    let mut removable = Vec::new();
    for j in 0..BANK_LINES {
        let y = y0 + j;
        // `removable` is the region the answer may write to, not the region this
        // answer happened to fill. The distinction is the whole family: `blank`
        // observes every cell it does not blank, so listing only the built cells
        // would leave an unbuilt line *given* as empty — the conditioning would
        // spell out the line count and the ambiguity would evaporate. Masking
        // the region instead asks the model the question we mean to ask: how
        // many lines belong here?
        removable.extend([
            grid.idx(x0 + 1, y),
            grid.idx(x0 + 2, y),
            grid.idx(x0 + 3, y),
        ]);
        // Every line above the sink's own row hands off to a belt running down
        // the column into the shared sink.
        if y != sink.1 {
            removable.push(grid.idx(x0 + 4, y));
        }
        if j < BANK_LINES - lines {
            continue;
        }
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
            y,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                item: recipe,
                misc: Misc::None,
            },
        );
        grid.set(
            x0 + 3,
            y,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        if y != sink.1 {
            grid.set(x0 + 4, y, Cell::belt(Direction::South));
        }
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
        assert!(LessonKind::all()
            .iter()
            .filter(|k| k.is_ambiguous())
            .all(|k| *k == LessonKind::AssemblerBank));
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

        for recipe in Item::craftable() {
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
}
