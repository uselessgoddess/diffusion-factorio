//! Procedural generation of known-correct factories ("lessons"), and blanking
//! them into (partial, solution) training pairs.
//!
//! This mirrors the reference project's data strategy
//! (`factorion_rs/src/factory_gen.rs`): each lesson is a distinct layout
//! pattern built by construction and verified functional (`sim::item_reaches_sink`),
//! then a subset of *removable* entities is blanked out. Source/sink/recipe
//! anchors may be `protected` for partial-inpainting examples. Production
//! training conditions only on source/sink task anchors; answer cells are never
//! revealed through this bookkeeping.
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
    /// The same craft, but nothing is stamped: obstacles, source, and sink form
    /// a random visible task; a deterministic solver then chooses the machine
    /// pose and *routes* the belts. The only machine lesson whose task space
    /// does not run out — see [`gen_assembler_chaos`].
    AssemblerChaos,
    /// Ingredient sources feed one assembler directly. Covers every recipe,
    /// including iron plate + copper cable → green circuit without an
    /// unnecessary cable assembler.
    DirectRecipe,
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
    /// **One** source, split by a splitter, feeding up to [`SHARED_LINES`]
    /// machines. The only family that shares an input line, and the only one that
    /// places a splitter at all — see [`gen_shared_line`].
    SharedLine,
}

impl LessonKind {
    pub fn all() -> &'static [LessonKind] {
        &[
            LessonKind::MoveOneItem,
            LessonKind::MoveOneItemChaos,
            LessonKind::AssemblerLine,
            LessonKind::AssemblerChaos,
            LessonKind::DirectRecipe,
            LessonKind::UndergroundCross,
            LessonKind::AssemblerBank,
            LessonKind::CircuitLine,
            LessonKind::SharedLine,
        ]
    }
    pub fn name(self) -> &'static str {
        match self {
            LessonKind::MoveOneItem => "MOVE_ONE_ITEM",
            LessonKind::MoveOneItemChaos => "MOVE_ONE_ITEM_CHAOS",
            LessonKind::AssemblerLine => "ASSEMBLER_LINE",
            LessonKind::AssemblerChaos => "ASSEMBLER_CHAOS",
            LessonKind::DirectRecipe => "DIRECT_RECIPE",
            LessonKind::UndergroundCross => "UNDERGROUND_CROSS",
            LessonKind::AssemblerBank => "ASSEMBLER_BANK",
            LessonKind::CircuitLine => "CIRCUIT_LINE",
            LessonKind::SharedLine => "SHARED_LINE",
        }
    }

    /// The smallest canvas this family fits on, as width × height.
    ///
    /// Each generator already refuses a canvas it cannot fit, so this is not what
    /// makes the curriculum correct — it is what keeps a caller from asking for
    /// a lesson that can never be built and burning [`generate`]'s whole retry
    /// budget discovering it. It lives here because the dimensions do; asked
    /// from `train.rs` it was a guess, and a wrong one (an assembler line is
    /// [`LINE_W`] columns wide, and was listed as needing five).
    ///
    /// Width and height are stated separately because the lessons are not
    /// square. This used to collapse to one number by taking the larger of the
    /// two, which is what a square curriculum forces and what quietly cost the
    /// most: a [`LessonKind::CircuitLine`] is 11×5 and was billed as needing 11
    /// rows it never touches — so on the 13×9 canvas the issue infers on, the
    /// only chain lesson in the curriculum was ruled out by six rows of nothing.
    pub fn min_canvas(self) -> Canvas {
        match self {
            LessonKind::MoveOneItem | LessonKind::MoveOneItemChaos => Canvas::square(3),
            LessonKind::AssemblerLine => Canvas::new(LINE_W, LINE_H),
            // The machine and its two inserters need 5 across in the worst case;
            // the rest is room for the router to have somewhere to route.
            LessonKind::AssemblerChaos => Canvas::square(7),
            LessonKind::DirectRecipe => Canvas::new(DIRECT_W, DIRECT_H),
            // One row: `S b d # u b K`.
            LessonKind::UndergroundCross => Canvas::new(7, 1),
            LessonKind::AssemblerBank => Canvas::new(LINE_W, LINE_H * BANK_LINES),
            LessonKind::CircuitLine => Canvas::new(CIRCUIT_W, CIRCUIT_H),
            LessonKind::SharedLine => Canvas::new(SHARED_W, SHARED_H),
        }
    }

    /// Can this family be built on `canvas` at all?
    pub fn fits(self, canvas: Canvas) -> bool {
        let need = self.min_canvas();
        canvas.width >= need.width && canvas.height >= need.height
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
        matches!(
            self,
            LessonKind::AssemblerBank | LessonKind::CircuitLine | LessonKind::SharedLine
        )
    }
}

/// Parallel lines an [`LessonKind::AssemblerBank`] scaffold offers.
pub const BANK_LINES: usize = 3;

/// Shortest side the default curriculum draws canvases from.
///
/// Nine, because [`LessonKind::AssemblerBank`] is nine rows tall and dropping
/// the bank would cost an ambiguous family. Everything narrower is still
/// *generable* — the generators only refuse what does not fit — it is just not
/// in the default pool.
pub const DEFAULT_CANVAS_MIN: usize = 9;

/// Longest side the default curriculum draws canvases from.
///
/// Fifteen, and the ceiling is the model rather than the generator: a `ResBlock`
/// tower of the default depth sees `2 * blocks + 1 = 13` cells
/// (`model::tests::the_default_tower_can_see_across_the_grids_we_train_on`), so
/// past roughly this width the far corners are joined only by the pooled global
/// context. Raising it is a config change on both sides, which is the point.
pub const DEFAULT_CANVAS_MAX: usize = 15;

/// The shape of the world a lesson is generated into.
///
/// Every lesson used to be built on `Grid::new(size, size)`, so a square canvas
/// was the only shape the model was ever shown — while the issue runs inference
/// on 13×9, and `experiments/grid_shape` measures what that costs: the same
/// task, unchanged, drops from 3.118 items/s on 11×11 to 0.478 on 13×9.
///
/// The alternative was to keep generating squares and pad them into the
/// rectangle at a random offset, which is cheaper to write and strictly worse to
/// learn from, for two reasons that `experiments/canvas_curriculum` measures:
///
/// * **It cannot show the lessons that matter.** A square lesson of side `s`
///   padded into `w`×`h` needs `s <= min(w, h)`, so 13×9 admits only lessons
///   with a min side of 9 — which excludes [`LessonKind::CircuitLine`] and
///   [`LessonKind::SharedLine`] outright, the only chain and the only splitter
///   in the curriculum. Generated natively they need 11×5 and 11×7 and fit 13×9
///   with room over. Padding drops exactly the compositional families the
///   "invent new solutions" goal rests on.
/// * **The pad region is a giveaway.** It is provably empty in every label, so
///   what the model learns from it is "the answer lives inside a square
///   sub-region" — the opposite of using the space it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
}

impl Canvas {
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub const fn square(side: usize) -> Self {
        Self::new(side, side)
    }

    pub const fn area(self) -> usize {
        self.width * self.height
    }

    /// An empty grid of this shape.
    pub fn grid(self) -> Grid {
        Grid::new(self.width, self.height)
    }

    /// Every canvas whose sides both fall in `min..=max`.
    ///
    /// This is what a curriculum is handed instead of one number. It is a
    /// *pool*, not a schedule: a shape is drawn per batch (`GridBatch` is one
    /// tensor and cannot hold a ragged batch), so over a run the model sees
    /// every shape in it and no cell position is ever reliably empty.
    pub fn pool(min: usize, max: usize) -> Vec<Canvas> {
        (min..=max)
            .flat_map(|height| (min..=max).map(move |width| Canvas::new(width, height)))
            .collect()
    }
}

impl std::fmt::Display for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

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

/// Generate a functional factory for `kind` on `canvas`, retrying with fresh
/// randomness until one validates. Deterministic in `seed`.
pub fn generate(kind: LessonKind, canvas: Canvas, seed: u64) -> Option<Sample> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let budget = (canvas.area() * 40).max(500);
    for _ in 0..budget {
        let built = match kind {
            LessonKind::MoveOneItem => gen_move_one_item(canvas, &mut rng, false),
            LessonKind::MoveOneItemChaos => gen_move_one_item(canvas, &mut rng, true),
            LessonKind::AssemblerLine => gen_assembler_line(canvas, &mut rng),
            LessonKind::AssemblerChaos => gen_assembler_chaos(canvas, &mut rng),
            LessonKind::DirectRecipe => gen_direct_recipe(canvas, &mut rng),
            LessonKind::UndergroundCross => gen_underground_cross(canvas, &mut rng),
            LessonKind::AssemblerBank => gen_assembler_bank(canvas, &mut rng),
            LessonKind::CircuitLine => gen_circuit_line(canvas, &mut rng),
            LessonKind::SharedLine => gen_shared_line(canvas, &mut rng),
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

fn random_cell(canvas: Canvas, rng: &mut ChaCha8Rng) -> (usize, usize) {
    (
        rng.gen_range(0..canvas.width),
        rng.gen_range(0..canvas.height),
    )
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

/// Pick the one assembler location implied by a visible chaos task.
///
/// The source, sink and obstacles are the model's conditioning. Choosing the
/// machine with fresh randomness after those are fixed would make its location
/// an unobservable label: the same task could demand any free 3x3 footprint.
/// Prefer the free footprint whose centre minimizes the total source-to-sink
/// detour, with coordinates breaking ties, so the answer is both useful and a
/// deterministic function of what the model sees.
fn canonical_assembler_anchor(
    task: &Grid,
    source: (usize, usize),
    sink: (usize, usize),
) -> Option<(usize, usize)> {
    if task.width < 3 || task.height < 3 {
        return None;
    }
    (0..=task.height - 3)
        .flat_map(|y| (0..=task.width - 3).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            (x..x + 3).all(|tx| {
                (y..y + 3).all(|ty| !task.is_obstacle(tx, ty) && task.anchor_at(tx, ty).is_none())
            })
        })
        .min_by_key(|&(x, y)| {
            let centre = (x + 1, y + 1);
            (manhattan(source, centre) + manhattan(centre, sink), y, x)
        })
}

/// BFS shortest path over free cells (4-connected), avoiding obstacles and
/// occupied cells. `start` and `goal` are always passable endpoints. Returns the
/// path including both endpoints, or `None`.
///
/// `avoid` is for cells that are free but must not be built on. The caller that
/// needs it is [`gen_assembler_chaos`], keeping its belts off the machine's
/// faces — see there for why a belt allowed to hug a machine breaks the graded
/// simulator.
fn bfs_path(
    grid: &Grid,
    start: (usize, usize),
    goal: (usize, usize),
    avoid: &[(usize, usize)],
) -> Option<Vec<(usize, usize)>> {
    let passable = |x: usize, y: usize| -> bool {
        if (x, y) == start || (x, y) == goal {
            return true;
        }
        if avoid.contains(&(x, y)) {
            return false;
        }
        // `anchor_at`, not `is_empty`: a multi-tile entity stores only its
        // top-left anchor and leaves the other footprint tiles `Empty` but
        // claimed (see `world.rs`). `is_empty` reads those as free and routes a
        // belt straight through the machine. Equivalent for the 1x1 callers this
        // started with, which is exactly why it was safe until it wasn't.
        !grid.is_obstacle(x, y) && grid.anchor_at(x, y).is_none()
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

fn gen_move_one_item(canvas: Canvas, rng: &mut ChaCha8Rng, chaos: bool) -> Option<Sample> {
    if !LessonKind::MoveOneItem.fits(canvas) {
        return None;
    }
    let mut grid = canvas.grid();

    if chaos {
        // Sprinkle a few obstacles (~10% of cells).
        let n_obstacles = (canvas.area() / 10).max(1);
        for _ in 0..n_obstacles {
            let (x, y) = random_cell(canvas, rng);
            grid.set_obstacle(x, y, true);
        }
    }

    let source = random_cell(canvas, rng);
    if grid.is_obstacle(source.0, source.1) {
        return None;
    }
    let sink = random_cell(canvas, rng);
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

    let path = bfs_path(&grid, source, sink, &[])?;
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
fn gen_assembler_line(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if !LessonKind::AssemblerLine.fits(canvas) {
        return None;
    }
    let y0 = rng.gen_range(0..=(canvas.height - LINE_H));
    let x0 = rng.gen_range(0..=(canvas.width - LINE_W));
    // One source feeds one machine, so this lesson can only teach the recipes
    // that need one thing. The two-input recipes are [`LessonKind::CircuitLine`].
    let recipe = *Item::single_input_craftable().choose(rng).unwrap();
    let input_item = recipe.ingredients()[0].item;

    // The machine's middle row: the one the line runs along.
    let y = y0 + 1;
    let mut grid = canvas.grid();
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
    // Partial inpainting preserves the assembler; task-conditioned training
    // intentionally ignores this list and masks it with the rest of the answer.
    let protected = vec![grid.idx(x0, y), grid.idx(x0 + 2, y0), grid.idx(x0 + 6, y)];
    let removable = vec![grid.idx(x0 + 1, y), grid.idx(x0 + 5, y)];
    Some(Sample {
        kind: LessonKind::AssemblerLine,
        solution: grid,
        protected,
        removable,
    })
}

/// Width of a direct recipe: west source + inserter, 3×3 machine, output
/// inserter + sink.
const DIRECT_W: usize = 7;

/// Height of a direct recipe: two rows for an optional north feed plus the
/// machine's three rows.
const DIRECT_H: usize = 5;

/// Feed the ingredients already named by the task directly into one machine.
///
/// ```text
///   . . . S . . .       optional second ingredient
///   . . . i . . .
///   . . A A A . .
///   S i A A A i K       first ingredient → product
///   . . A A A . .
/// ```
///
/// [`LessonKind::CircuitLine`] intentionally teaches composition from copper
/// plate: one assembler makes cable and another consumes it. That is a
/// different task from a supplied-cable circuit build, where adding the cable
/// assembler is wrong. This family makes the recipe graph explicit at the task
/// boundary and samples every craftable product.
fn gen_direct_recipe(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if !LessonKind::DirectRecipe.fits(canvas) {
        return None;
    }
    let x0 = rng.gen_range(0..=(canvas.width - DIRECT_W));
    let y0 = rng.gen_range(0..=(canvas.height - DIRECT_H));
    let recipe = *Item::craftable().choose(rng).unwrap();
    let mut ingredients: Vec<Item> = recipe.ingredients().iter().map(|i| i.item).collect();
    ingredients.shuffle(rng);

    let machine = (x0 + 2, y0 + 2);
    let west_source = (x0, y0 + 3);
    let west_inserter = (x0 + 1, y0 + 3);
    let north_source = (x0 + 3, y0);
    let north_inserter = (x0 + 3, y0 + 1);
    let output_inserter = (x0 + 5, y0 + 3);
    let sink = (x0 + 6, y0 + 3);

    let mut grid = canvas.grid();
    grid.set(
        west_source.0,
        west_source.1,
        Cell {
            entity: Entity::Source,
            item: ingredients[0],
            ..Default::default()
        },
    );
    grid.set(west_inserter.0, west_inserter.1, inserter(Direction::East));
    if let Some(&second) = ingredients.get(1) {
        grid.set(
            north_source.0,
            north_source.1,
            Cell {
                entity: Entity::Source,
                item: second,
                ..Default::default()
            },
        );
        grid.set(
            north_inserter.0,
            north_inserter.1,
            inserter(Direction::South),
        );
    }
    grid.set(
        machine.0,
        machine.1,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: recipe,
            misc: Misc::None,
        },
    );
    grid.set(
        output_inserter.0,
        output_inserter.1,
        inserter(Direction::East),
    );
    grid.set(
        sink.0,
        sink.1,
        Cell {
            entity: Entity::Sink,
            item: recipe,
            ..Default::default()
        },
    );

    if !item_reaches_sink(&grid) {
        return None;
    }
    let protected: Vec<usize> = grid
        .cells
        .iter()
        .enumerate()
        .filter_map(|(i, c)| matches!(c.entity, Entity::Source | Entity::Sink).then_some(i))
        .collect();
    let mut removable = Vec::new();
    for y in y0..y0 + DIRECT_H {
        for x in x0..x0 + DIRECT_W {
            let i = grid.idx(x, y);
            if !protected.contains(&i) {
                removable.push(i);
            }
        }
    }
    Some(Sample {
        kind: LessonKind::DirectRecipe,
        solution: grid,
        protected,
        removable,
    })
}

/// Obstacles to scatter, as a fraction of the board. Matches the ~10% budget
/// [`gen_move_one_item`] uses in chaos mode.
const CHAOS_OBSTACLES: usize = 10;

/// The same craft as [`gen_assembler_line`], with nothing stamped:
///
/// ```text
///   . # . . b b K       S  source        A  assembler (3x3, anchored)
///   S b b . b # .       K  sink          i  inserter
///   . . b . i . .       b  belt          #  obstacle
///   # . b b A A A
///   . . . . A A A
///   . . i . A A A
///   . . b b b b .
/// ```
///
/// Why this family exists: `task_space` counts every other machine lesson at a
/// handful of distinct layouts — [`LessonKind::AssemblerLine`] has **2**, one per
/// recipe — because they place a fixed template at
/// `rng.gen_range(0..=(size - W))` and vary nothing else. The rest of their
/// apparent task space is the same picture at another offset, and the denoiser
/// is `same`-padded convolution end to end, so translation is the one variation
/// it already generalizes over for free. A bigger board multiplies the offsets
/// and not the layouts; see `docs/ROADMAP.md` bottleneck 0.
///
/// [`LessonKind::MoveOneItemChaos`] is the family that does not have this
/// problem (200,000 distinct layouts from 200,000 seeds, at every size), and the
/// reason is that it does not stamp: obstacles go into the conditioning plane
/// and the belts are derived by BFS *through* them, so the label is a function
/// of a world the model can see. This applies that to the craft:
///
/// * obstacles are scattered first, and the router has to respect them;
/// * source and sink land anywhere they fit;
/// * the closest feasible machine footprint and its inserter faces are derived
///   deterministically from that visible task;
/// * the belts are whatever BFS finds, so the answer depends on the obstacles
///   rather than ignoring them.
///
/// Placing obstacles that the answer *ignores* would be worse than useless: it
/// inflates every distinctness count while teaching nothing. What makes them
/// count is that they are in the path.
fn gen_assembler_chaos(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if !LessonKind::AssemblerChaos.fits(canvas) {
        return None;
    }
    let mut grid = canvas.grid();
    for _ in 0..(canvas.area() / CHAOS_OBSTACLES).max(1) {
        let (x, y) = random_cell(canvas, rng);
        grid.set_obstacle(x, y, true);
    }

    let recipe = *Item::single_input_craftable().choose(rng).unwrap();
    let input_item = recipe.ingredients()[0].item;

    // Pose the visible task first. Everything below this point is a
    // deterministic solver: varied tasks still produce varied layouts, but the
    // target no longer contains hidden RNG choices the denoiser cannot infer.
    let source = random_cell(canvas, rng);
    let sink = random_cell(canvas, rng);
    if source == sink || grid.is_obstacle(source.0, source.1) || grid.is_obstacle(sink.0, sink.1) {
        return None;
    }
    grid.set(
        source.0,
        source.1,
        Cell {
            entity: Entity::Source,
            item: input_item,
            ..Default::default()
        },
    );
    grid.set(
        sink.0,
        sink.1,
        Cell {
            entity: Entity::Sink,
            item: recipe,
            ..Default::default()
        },
    );

    // The shortest useful free footprint is now implied by the task, rather
    // than sampled independently of it.
    let (ax, ay) = canonical_assembler_anchor(&grid, source, sink)?;
    grid.set(
        ax,
        ay,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: recipe,
            misc: Misc::None,
        },
    );

    // A port directly touching the machine would bypass one of the inserters
    // and change the task's production graph.
    let perimeter = grid.perimeter(ax, ay);
    if perimeter.contains(&source) || perimeter.contains(&sink) {
        return None;
    }

    // Two distinct free faces: one to load, one to unload. `perimeter` is in
    // anchor coordinates and already excludes the footprint itself. A corner of
    // the ring touches the machine only diagonally and no inserter can reach it,
    // so `dir_toward_footprint` rejects it here rather than three checks later.
    let faces: Vec<(usize, usize)> = perimeter
        .into_iter()
        .filter(|&(x, y)| !grid.is_obstacle(x, y) && grid.anchor_at(x, y).is_none())
        .filter(|&f| dir_toward_footprint(&grid, f, (ax, ay)).is_some())
        .collect();

    // Enumerate the machine's input/output faces deterministically, preferring
    // the shortest combined route. Trying alternatives is still deterministic;
    // it only handles a blocked first choice.
    let mut pairs = Vec::new();
    for &load in &faces {
        let Some(into_machine) = dir_toward_footprint(&grid, load, (ax, ay)) else {
            continue;
        };
        let Some(pickup) = step(load, into_machine.opposite(), canvas) else {
            continue;
        };
        for &unload in &faces {
            if unload == load {
                continue;
            }
            let Some(away) = dir_toward_footprint(&grid, unload, (ax, ay)).map(Direction::opposite)
            else {
                continue;
            };
            let Some(drop) = step(unload, away, canvas) else {
                continue;
            };
            pairs.push((load, unload, into_machine, away, pickup, drop));
        }
    }
    pairs.sort_by_key(|&(load, unload, _, _, pickup, drop)| {
        (
            manhattan(source, pickup) + manhattan(drop, sink),
            load.1,
            load.0,
            unload.1,
            unload.0,
        )
    });

    for (load, unload, into_machine, away, pickup, drop) in pairs {
        let mut candidate = grid.clone();
        let keep_clear: Vec<(usize, usize)> = faces
            .iter()
            .copied()
            .filter(|&f| f != load && f != unload)
            .collect();

        if [pickup, drop]
            .into_iter()
            .any(|c| candidate.is_obstacle(c.0, c.1) || candidate.anchor_at(c.0, c.1).is_some())
            || neighbours(source, canvas).contains(&drop)
        {
            continue;
        }
        let named = [source, sink, load, unload, pickup, drop];
        if (0..named.len()).any(|i| named[i + 1..].contains(&named[i])) {
            continue;
        }

        candidate.set(
            load.0,
            load.1,
            Cell {
                entity: Entity::Inserter,
                direction: into_machine,
                ..Default::default()
            },
        );
        candidate.set(
            unload.0,
            unload.1,
            Cell {
                entity: Entity::Inserter,
                direction: away,
                ..Default::default()
            },
        );
        candidate.set(pickup.0, pickup.1, Cell::belt(into_machine));
        candidate.set(drop.0, drop.1, Cell::belt(away));

        let mut removable = vec![
            candidate.idx(load.0, load.1),
            candidate.idx(unload.0, unload.1),
            candidate.idx(pickup.0, pickup.1),
            candidate.idx(drop.0, drop.1),
        ];
        let Some((inbound, _)) = belt_run(&mut candidate, source, pickup, &keep_clear) else {
            continue;
        };
        removable.extend(inbound);

        // The output route must not brush the unlimited input source: doing so
        // replaces the finite crafted output in the throughput simulation.
        let mut gear_keep_clear = keep_clear;
        gear_keep_clear.extend(neighbours(source, canvas));
        let Some((outbound, out_dir)) = belt_run(&mut candidate, drop, sink, &gear_keep_clear)
        else {
            continue;
        };
        removable.extend(outbound);
        candidate.set(drop.0, drop.1, Cell::belt(out_dir));

        if !item_reaches_sink(&candidate) {
            continue;
        }
        let protected = vec![
            candidate.idx(source.0, source.1),
            candidate.idx(sink.0, sink.1),
            candidate.idx(ax, ay),
        ];
        return Some(Sample {
            kind: LessonKind::AssemblerChaos,
            solution: candidate,
            protected,
            removable,
        });
    }

    None
}

/// The up-to-four cells orthogonally adjacent to `pos` and still on the board.
///
/// Which is exactly the set a 1x1 entity offers its output to, so it is also the
/// set a belt has to stay out of to avoid being offered something.
fn neighbours(pos: (usize, usize), canvas: Canvas) -> Vec<(usize, usize)> {
    [
        Direction::North,
        Direction::South,
        Direction::East,
        Direction::West,
    ]
    .into_iter()
    .filter_map(|d| step(pos, d, canvas))
    .collect()
}

/// One step from `pos` along `d`, or `None` if that leaves the board.
fn step(pos: (usize, usize), d: Direction, canvas: Canvas) -> Option<(usize, usize)> {
    let (dx, dy) = d.delta();
    let (x, y) = (pos.0 as i32 + dx, pos.1 as i32 + dy);
    (x >= 0 && y >= 0 && (x as usize) < canvas.width && (y as usize) < canvas.height)
        .then_some((x as usize, y as usize))
}

/// The direction from a perimeter cell into the footprint anchored at `anchor`.
/// `None` for a cell that only touches the footprint diagonally.
fn dir_toward_footprint(
    grid: &Grid,
    from: (usize, usize),
    anchor: (usize, usize),
) -> Option<Direction> {
    let body = grid.footprint_at(anchor.0, anchor.1);
    [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ]
    .into_iter()
    .find(|d| {
        let (dx, dy) = d.delta();
        body.contains(&(from.0 as i32 + dx, from.1 as i32 + dy))
    })
}

/// Lay a belt run from `start` to `goal`, exclusive of both, each belt facing
/// the next cell along the route. Returns the indices it wrote, and the
/// direction the route leaves `start` by.
///
/// `start` and `goal` are the *endpoints that already exist* — a source and an
/// inserter, say. `bfs_path` treats them as passable and everything occupied as
/// not, so the run threads between what is already on the board.
///
/// The returned direction is what lets a caller point an inserter at the route
/// instead of guessing where the route will go. When `start` and `goal` are
/// already adjacent there are no belts to write, and it is the direction from
/// one to the other — still the way out of `start`, still correct to face.
fn belt_run(
    grid: &mut Grid,
    start: (usize, usize),
    goal: (usize, usize),
    avoid: &[(usize, usize)],
) -> Option<(Vec<usize>, Direction)> {
    let path = bfs_path(grid, start, goal, avoid)?;
    // `None` rather than a panic when `start == goal`: a caller that routed a
    // cell to itself has a bug, but the retry loop in `generate` is the right
    // place to notice, not an index out of bounds.
    let out = dir_between(start, *path.get(1)?);
    let interior = &path[1..path.len() - 1];
    let mut idx = Vec::with_capacity(interior.len());
    for (i, &pos) in interior.iter().enumerate() {
        grid.set(pos.0, pos.1, Cell::belt(dir_between(pos, path[i + 2])));
        idx.push(grid.idx(pos.0, pos.1));
    }
    Some((idx, out))
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
fn gen_assembler_bank(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    // Lines stack back to back: three rows each, and no gap between them, so a
    // bank of three is nine rows tall. The columns are a line's own, plus one
    // for the belt that merges every line into the shared sink.
    let bank = LessonKind::AssemblerBank.min_canvas();
    if !LessonKind::AssemblerBank.fits(canvas) {
        return None;
    }
    let y0 = rng.gen_range(0..=(canvas.height - bank.height));
    let x0 = rng.gen_range(0..=(canvas.width - bank.width));
    // Every line here is fed by one source, so — as in [`gen_assembler_line`] —
    // only the single-input recipes fit.
    let recipe = *Item::single_input_craftable().choose(rng).unwrap();
    let input_item = recipe.ingredients()[0].item;

    // Line `j` is anchored at `machine(j)` and runs along `row(j)`, its middle.
    let machine = |j: usize| (x0 + 2, y0 + LINE_H * j);
    let row = |j: usize| y0 + LINE_H * j + 1;
    let column = x0 + bank.width - 1;
    let sink = (column, row(BANK_LINES - 1));

    let mut grid = canvas.grid();
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
    for y in y0..y0 + bank.height {
        for x in x0..x0 + bank.width {
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
fn gen_circuit_line(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if !LessonKind::CircuitLine.fits(canvas) {
        return None;
    }
    let x0 = rng.gen_range(0..=(canvas.width - CIRCUIT_W));
    let y0 = rng.gen_range(0..=(canvas.height - CIRCUIT_H));

    // The row the line runs along: the machines' middle.
    let y = y0 + 3;
    let cable = (x0 + 2, y0 + 2);
    let circuit = (x0 + 6, y0 + 2);
    let iron_source = (x0 + 6, y0);
    let copper_source = (x0, y);
    let sink = (x0 + CIRCUIT_W - 1, y);

    let mut grid = canvas.grid();
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

fn gen_underground_cross(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    // Horizontal line with an obstacle wall the belts tunnel under:
    // S b d # u b K   (wall at the '#').
    if !LessonKind::UndergroundCross.fits(canvas) {
        return None;
    }
    let y = rng.gen_range(0..canvas.height);
    let x0 = rng.gen_range(0..=(canvas.width - 7));
    let item = *[Item::IronPlate, Item::CopperPlate].choose(rng).unwrap();

    let mut grid = canvas.grid();
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

/// Machines a [`LessonKind::SharedLine`] scaffold can feed from its one source.
///
/// Two, because a splitter has two outputs and this family is about the splitter.
/// Feeding four would mean a tree of three splitters and a much taller scaffold;
/// the concept — *one input line serves many machines* — is already the
/// difference between one and two.
pub const SHARED_LINES: usize = 2;

/// Columns a [`LessonKind::SharedLine`] occupies: source, its belt, the splitter,
/// the branch belt, the loading inserter, the machine's three, the unloading
/// inserter, the output belt, the collector column.
const SHARED_W: usize = 11;

/// Rows it occupies: two machines three tall, one row apart.
const SHARED_H: usize = 7;

/// Up to [`SHARED_LINES`] machines fed from **one** source through a splitter:
///
/// ```text
///   S  >  =  >  i  A  A  A  i  >  v
///   .  .  =  v  .  A  A  A  .  .  v
///   .  .  .  v  .  A  A  A  .  .  v
///   .  .  .  v  .  .  .  .  .  .  v
///   .  .  .  >  i  A  A  A  i  >  K
///   .  .  .  .  .  A  A  A  .  .  .
///   .  .  .  .  .  A  A  A  .  .  .
/// ```
///
/// [`gen_assembler_bank`] gives each of its three machines a source of its own.
/// No real factory looks like that — a real one runs **one** belt and divides it,
/// because input lines are the scarce thing and machines are not. This family is
/// that correction, and it is the only place in the curriculum where either half
/// of the idea appears:
///
/// * **One source, many machines.** The bank's scaffold answers "how many
///   machines" by counting the sources it was handed. Here the scaffold says
///   nothing about the count, because there is one source whatever the answer is.
/// * **A splitter.** `Entity::Splitter` has been in the vocabulary, the
///   simulator, the SVG and the blueprint exporter from the start, and no lesson
///   has ever placed one — `experiments/bus_tap` counts zero across every family.
///   The model has had a word it was never shown a use for.
///
/// The ambiguity is [`gen_assembler_bank`]'s, kept deliberately: **how many of
/// the two branches get built is up to the answer**, both branches deliver into
/// the same sink, and two deliver twice what one does. So the family has more
/// than one working answer and they are ordered — which is what gives
/// [`crate::throughput`] and [`crate::best_of_n`] something to do.
///
/// A one-branch answer needs no splitter at all and the splitter's dead output
/// costs nothing: [`crate::sim::flow_targets`] only names tiles something stands
/// on, so an unbuilt branch is not a successor and the splitter does not divide
/// its flow into the void.
///
/// Only single-input recipes fit, as in [`gen_assembler_line`]: there is one
/// source, so there is one ingredient.
fn gen_shared_line(canvas: Canvas, rng: &mut ChaCha8Rng) -> Option<Sample> {
    if !LessonKind::SharedLine.fits(canvas) {
        return None;
    }
    let x0 = rng.gen_range(0..=(canvas.width - SHARED_W));
    let y0 = rng.gen_range(0..=(canvas.height - SHARED_H));
    let recipe = *Item::single_input_craftable().choose(rng).unwrap();
    let input_item = recipe.ingredients()[0].item;

    // Branch `j` runs along `row(j)`: the top one shares the source's row, the
    // bottom one sits four rows down so the 3×3 machines clear each other.
    let row = |j: usize| y0 + 4 * j;
    let machine = |j: usize| (x0 + 5, row(j));
    let collector = x0 + SHARED_W - 1;
    let sink = (collector, row(SHARED_LINES - 1));

    let mut grid = canvas.grid();
    grid.set(
        x0,
        y0,
        Cell {
            entity: Entity::Source,
            item: input_item,
            ..Default::default()
        },
    );
    grid.set(
        sink.0,
        sink.1,
        Cell {
            entity: Entity::Sink,
            item: recipe,
            ..Default::default()
        },
    );
    let protected = vec![grid.idx(x0, y0), grid.idx(sink.0, sink.1)];

    // As in the bank: mask the whole rectangle, not the cells this answer
    // happened to fill. Observing an unbuilt branch as empty would spell the
    // branch count out in the conditioning and the ambiguity would evaporate.
    let mut removable = Vec::new();
    for y in y0..y0 + SHARED_H {
        for x in x0..x0 + SHARED_W {
            let i = grid.idx(x, y);
            if !protected.contains(&i) {
                removable.push(i);
            }
        }
    }

    let branches = rng.gen_range(1..=SHARED_LINES);

    // The head of the line: source, belt, and — only when the line is actually
    // divided — the splitter. An east-facing splitter is 1×2, anchored on the
    // source's row and reaching one row down, and each of its two tiles pushes
    // east independently. That is the whole entity: one belt in, two belts out.
    grid.set(x0 + 1, y0, Cell::belt(Direction::East));
    if branches > 1 {
        grid.set(
            x0 + 2,
            y0,
            Cell {
                entity: Entity::Splitter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        // The lower output turns down the branch column.
        grid.set(x0 + 3, y0 + 1, Cell::belt(Direction::South));
        for y in y0 + 2..row(1) {
            grid.set(x0 + 3, y, Cell::belt(Direction::South));
        }
    } else {
        grid.set(x0 + 2, y0, Cell::belt(Direction::East));
    }

    for j in 0..branches {
        let y = row(j);
        // Belt into the machine's loading inserter, machine, unloading inserter.
        grid.set(x0 + 3, y, Cell::belt(Direction::East));
        grid.set(
            x0 + 4,
            y,
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
            x0 + 8,
            y,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        grid.set(x0 + 9, y, Cell::belt(Direction::East));
    }
    // One column collects every branch's output into the shared sink, which sits
    // at the bottom branch's row: the bottom branch hands off to it directly, the
    // top one belts down the column. The column is laid whatever the branch
    // count, because a one-branch answer is only the top branch and would
    // otherwise deliver into an empty tile — `item_reaches_sink` would reject it
    // and `generate`'s retry loop would silently redraw until it got two, leaving
    // a family that claims to be ambiguous and never once is.
    for y in y0..sink.1 {
        grid.set(collector, y, Cell::belt(Direction::South));
    }

    if !item_reaches_sink(&grid) {
        return None;
    }
    Some(Sample {
        kind: LessonKind::SharedLine,
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
                let Some(sample) = generate(LessonKind::AssemblerBank, Canvas::square(11), seed)
                else {
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
            .all(|k| matches!(
                k,
                LessonKind::AssemblerBank | LessonKind::CircuitLine | LessonKind::SharedLine
            )));
    }

    /// Ambiguity alone is not enough: if every answer delivered the same rate,
    /// ranking them would still be a coin flip. The answers have to be *unequal*,
    /// and by a real margin — that margin is the gradient Best-of-N climbs and
    /// the one `beat_original` reports.
    #[test]
    fn the_bank_answers_are_not_equally_good() {
        let mut by_rate: HashMap<String, Vec<f64>> = HashMap::new();
        for seed in 0..2_000u64 {
            let Some(sample) = generate(LessonKind::AssemblerBank, Canvas::square(11), seed) else {
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
            let Some(sample) = generate(LessonKind::AssemblerBank, Canvas::square(11), seed) else {
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
                if let Some(s) = generate(kind, Canvas::square(11), seed) {
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

    /// The real failure report asks for a green circuit from iron plates and
    /// already-made copper cable. `CIRCUIT_LINE` cannot teach that contract: it
    /// starts from copper plate and therefore labels a two-assembler chain as
    /// correct. The direct family must cover the one-machine version as well as
    /// every other recipe.
    #[test]
    fn direct_recipe_covers_every_recipe_and_one_machine_circuits() {
        let mut recipes = HashSet::new();
        let mut circuit_cases = 0;
        for seed in 0..200u64 {
            let sample = generate(LessonKind::DirectRecipe, Canvas::new(13, 9), seed)
                .expect("direct recipe must fit the inference canvas");
            let assemblers: Vec<Cell> = sample
                .solution
                .cells
                .iter()
                .copied()
                .filter(|c| c.entity == Entity::Assembler)
                .collect();
            assert_eq!(assemblers.len(), 1);
            recipes.insert(assemblers[0].item as u8);
            assert!(item_reaches_sink(&sample.solution));

            if assemblers[0].item == Item::GreenCircuit {
                circuit_cases += 1;
                let sources: HashSet<u8> = sample
                    .solution
                    .cells
                    .iter()
                    .filter(|c| c.entity == Entity::Source)
                    .map(|c| c.item as u8)
                    .collect();
                assert_eq!(
                    sources,
                    HashSet::from([Item::IronPlate as u8, Item::CopperCable as u8])
                );
                assert!(sample.protected.iter().all(|&i| matches!(
                    sample.solution.cells[i].entity,
                    Entity::Source | Entity::Sink
                )));
            }
        }
        assert!(circuit_cases > 0);
        assert_eq!(
            recipes,
            Item::craftable().into_iter().map(|i| i as u8).collect()
        );
    }

    /// An inserter that pushes into an empty tile is decorative, and a factory
    /// that needs it to be decorative is not buildable.
    ///
    /// [`gen_assembler_chaos`] shipped this bug and the simulator waved it
    /// through: it picked the unloading inserter's direction from the machine's
    /// geometry, then let BFS route the belt out of some *other* face, leaving
    /// the inserter aimed at nothing. `item_reaches_sink` still said yes, because
    /// `flow_targets` lets an assembler offer to its whole perimeter, and the
    /// first belt of the run was standing on that perimeter — so the machine fed
    /// the belt directly and the inserter was scenery. Factorio has no such
    /// shortcut, and a model trained on it would learn to place inserters that
    /// mean nothing. The simulator cannot catch this, so a test has to.
    #[test]
    fn every_inserter_pushes_into_something_real() {
        for &kind in LessonKind::all() {
            for seed in 0..60u64 {
                let Some(s) = generate(kind, Canvas::square(11), seed) else {
                    continue;
                };
                let g = &s.solution;
                for y in 0..g.height {
                    for x in 0..g.width {
                        if g.get(x, y).entity != Entity::Inserter {
                            continue;
                        }
                        let (dx, dy) = g.get(x, y).direction.delta();
                        let (tx, ty) = (x as i32 + dx, y as i32 + dy);
                        assert!(
                            tx >= 0
                                && ty >= 0
                                && (tx as usize) < g.width
                                && (ty as usize) < g.height,
                            "{}: inserter at ({x},{y}) faces off the board",
                            kind.name()
                        );
                        assert!(
                            g.anchor_at(tx as usize, ty as usize).is_some(),
                            "{}: inserter at ({x},{y}) pushes into an empty tile (seed {seed})",
                            kind.name()
                        );
                    }
                }
            }
        }
    }

    /// The measurement that justifies this whole family, small enough to run in
    /// CI: `experiments/task_space` counts the *answers* each lesson teaches —
    /// the cells the model must fill, with translation collapsed, because the
    /// denoiser is `same`-padded convolution and slides for free.
    ///
    /// [`LessonKind::AssemblerLine`] scores **two**, one per recipe: production
    /// training now masks the assembler as well as the belts and inserters. A
    /// 5,000-step run still draws each template thousands of times. That is the
    /// bottleneck, and it is exactly as bad on a bigger board — a wider grid
    /// buys offsets, which cost compute and teach nothing (`docs/ROADMAP.md`
    /// bottleneck 0).
    #[test]
    fn the_chaos_family_teaches_varied_answers_and_the_templated_one_teaches_two() {
        /// The answer alone, normalized to its own bounding box.
        fn answer_shape(s: &Sample) -> String {
            let g = &s.solution;
            let cells: Vec<(usize, usize, Cell)> = s
                .solution
                .cells
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    !c.is_empty() && !matches!(c.entity, Entity::Source | Entity::Sink)
                })
                .map(|(i, &c)| (i % g.width, i / g.width, c))
                .collect();
            let min_x = cells.iter().map(|&(x, _, _)| x).min().unwrap();
            let min_y = cells.iter().map(|&(_, y, _)| y).min().unwrap();
            let mut keys: Vec<String> = cells
                .iter()
                .map(|&(x, y, c)| {
                    format!(
                        "{},{}:{}:{}:{}:{}",
                        x - min_x,
                        y - min_y,
                        c.entity as u8,
                        c.direction as u8,
                        c.item as u8,
                        c.misc as u8
                    )
                })
                .collect();
            keys.sort();
            keys.join(";")
        }

        let shapes = |kind| -> usize {
            (0..200u64)
                .filter_map(|seed| generate(kind, Canvas::square(11), seed))
                .map(|s| answer_shape(&s))
                .collect::<HashSet<_>>()
                .len()
        };

        assert_eq!(
            shapes(LessonKind::AssemblerLine),
            2,
            "the templated line is supposed to teach exactly two recipe answers -- if this \
             moved, the premise of ASSEMBLER_CHAOS changed and the docs are stale"
        );
        let chaos = shapes(LessonKind::AssemblerChaos);
        assert!(
            chaos > 150,
            "ASSEMBLER_CHAOS gave only {chaos} distinct answers in 200 seeds; it is \
             stamping a template again"
        );
    }

    #[test]
    fn assembler_chaos_machine_is_determined_by_the_visible_task() {
        for canvas in Canvas::pool(DEFAULT_CANVAS_MIN, DEFAULT_CANVAS_MAX) {
            for seed in 0..4u64 {
                let sample = generate(LessonKind::AssemblerChaos, canvas, seed)
                    .expect("chaos task should generate");
                let (task, _) = sample.blank_to_scaffold();
                let source = sample
                    .solution
                    .cells
                    .iter()
                    .position(|c| c.entity == Entity::Source)
                    .map(|i| (i % sample.solution.width, i / sample.solution.width))
                    .unwrap();
                let sink = sample
                    .solution
                    .cells
                    .iter()
                    .position(|c| c.entity == Entity::Sink)
                    .map(|i| (i % sample.solution.width, i / sample.solution.width))
                    .unwrap();
                let actual = sample
                    .solution
                    .cells
                    .iter()
                    .position(|c| c.entity == Entity::Assembler)
                    .map(|i| (i % sample.solution.width, i / sample.solution.width))
                    .unwrap();

                assert_eq!(
                    actual,
                    canonical_assembler_anchor(&task, source, sink).unwrap(),
                    "{canvas} seed {seed} teaches a hidden random machine location"
                );
                assert!(
                    throughput::score(&sample.solution) > 0.0,
                    "{canvas} seed {seed} routes but delivers nothing"
                );
            }
        }
    }

    #[test]
    fn blanking_masks_removable_cells() {
        let s = generate(LessonKind::MoveOneItem, Canvas::square(11), 7).expect("gen");
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
            let Some(s) = generate(kind, Canvas::square(11), 7) else {
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
        let s = generate(LessonKind::AssemblerLine, Canvas::square(11), 7).expect("gen");
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
            let Some(sample) = generate(LessonKind::CircuitLine, Canvas::square(11), seed) else {
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
            let Some(sample) = generate(LessonKind::CircuitLine, Canvas::square(11), seed) else {
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
        let sample = generate(LessonKind::CircuitLine, Canvas::square(11), 0).expect("gen");
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

    /// Every scaffold the generator writes has to be buildable and has to
    /// deliver, or the model is being trained on a lie. Same shape as
    /// [`every_circuit_line_actually_delivers_circuits`], and worth repeating
    /// here because this is the only family that places a splitter — an entity
    /// whose footprint rotates (1x2 facing east, 2x1 facing north) and whose flow
    /// divides, so there are two ways for it to be quietly wrong.
    #[test]
    fn every_shared_line_actually_delivers() {
        let mut built = 0;
        for seed in 0..200u64 {
            let Some(sample) = generate(LessonKind::SharedLine, Canvas::square(11), seed) else {
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
            assert_eq!(
                sample
                    .solution
                    .cells
                    .iter()
                    .filter(|c| c.entity == Entity::Source)
                    .count(),
                1,
                "seed {seed} has more than the one source that is the whole point"
            );
        }
        assert_eq!(built, 200, "the generator failed on some seeds");
    }

    /// The point of the family, as a number rather than a claim in a doc comment:
    /// **one** source feeds two machines through a splitter, and two machines
    /// deliver twice what one does.
    ///
    /// [`gen_assembler_bank`] gets the same doubling by handing the answer a
    /// second source. That is the thing this family exists to correct, so the
    /// test asserts the source count too — if the doubling ever came from extra
    /// input rather than from dividing the input there was, the lesson would have
    /// silently become the bank again.
    #[test]
    fn one_source_split_two_ways_delivers_twice_what_one_branch_does() {
        let mut by_recipe: HashMap<(u8, usize), f64> = HashMap::new();
        for seed in 0..2_000u64 {
            let Some(sample) = generate(LessonKind::SharedLine, Canvas::square(11), seed) else {
                continue;
            };
            let branches = sample
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
                .expect("always has a sink")
                .item as u8;
            let rate = throughput::score(&sample.solution);
            if let Some(previous) = by_recipe.insert((recipe, branches), rate) {
                assert!(
                    (previous - rate).abs() < 1e-9,
                    "{branches} branches of recipe {recipe} delivered {previous} and {rate}"
                );
            }
            // A one-branch answer needs no splitter, and only a one-branch answer
            // may go without: an unsplit line cannot reach the second machine.
            let splitters = sample
                .solution
                .cells
                .iter()
                .filter(|c| c.entity == Entity::Splitter)
                .count();
            assert_eq!(
                splitters,
                usize::from(branches > 1),
                "seed {seed} built {branches} branches off {splitters} splitters"
            );
        }

        let recipes: HashSet<u8> = by_recipe.keys().map(|&(r, _)| r).collect();
        assert_eq!(
            recipes.len(),
            Item::single_input_craftable().len(),
            "not every recipe was drawn"
        );
        for recipe in recipes {
            let (Some(&one), Some(&two)) =
                (by_recipe.get(&(recipe, 1)), by_recipe.get(&(recipe, 2)))
            else {
                panic!("recipe {recipe} was never drawn at both branch counts");
            };
            assert!(
                (two - one * 2.0).abs() < 1e-9,
                "recipe {recipe}: splitting the line took {one}/s to {two}/s, not to {}/s",
                one * 2.0
            );
        }
    }

    /// `Entity::Splitter` has been in the vocabulary, the simulator, the SVG
    /// renderer and the blueprint exporter since the first commit, and until
    /// [`gen_shared_line`] no lesson had ever placed one: `experiments/bus_tap`
    /// counted zero across every family at 200 seeds each. The model had a word
    /// it was never shown a use for, and a word never seen in training is a word
    /// never drawn at inference.
    ///
    /// The test is here so that stays fixed. If someone deletes this family, the
    /// vocabulary quietly develops a hole again and nothing else in the suite
    /// would say so.
    #[test]
    fn some_lesson_teaches_every_entity_the_vocabulary_has() {
        let mut seen: HashSet<u8> = HashSet::new();
        for &kind in LessonKind::all() {
            for seed in 0..50u64 {
                let Some(sample) = generate(kind, Canvas::square(13), seed) else {
                    continue;
                };
                seen.extend(sample.solution.cells.iter().map(|c| c.entity as u8));
            }
        }
        let untaught: Vec<Entity> = (0..Entity::COUNT)
            .filter_map(Entity::from_id)
            .filter(|e| !seen.contains(&(*e as u8)))
            .collect();
        assert!(
            untaught.is_empty(),
            "no lesson ever places {untaught:?} -- the model cannot learn a word it never sees"
        );
    }

    /// The canvas the issue actually runs inference on.
    const INFERENCE: Canvas = Canvas::new(13, 9);

    /// The measurement behind [`Canvas`]'s decision, as an assertion.
    ///
    /// Padding a square lesson of side `s` into `w`×`h` needs `s <= min(w, h)`,
    /// so the widest square 13×9 admits is 9×9 — and both compositional families
    /// are wider than they are tall. Native generation asks each side separately
    /// and both fit with room over. This is the whole argument for paying for
    /// two-sided generation, so if someone collapses `min_canvas` back to one
    /// number this says which lessons it costs.
    #[test]
    fn padding_squares_would_drop_the_two_lessons_that_compose() {
        for kind in [LessonKind::CircuitLine, LessonKind::SharedLine] {
            let need = kind.min_canvas();
            assert!(
                kind.fits(INFERENCE),
                "{} needs {need} and does not fit {INFERENCE} natively",
                kind.name()
            );
            let largest_padded_square = Canvas::square(INFERENCE.width.min(INFERENCE.height));
            assert!(
                !kind.fits(largest_padded_square),
                "{} fits {largest_padded_square}, so padding would not have cost it \
                 -- this test no longer measures anything",
                kind.name()
            );
        }
    }

    /// Every family is buildable on the inference canvas, not merely admitted by
    /// [`LessonKind::fits`]. `fits` is a cheap arithmetic gate; the generators are
    /// generate-and-verify and can still come back empty, which would leave a
    /// lesson silently absent from a curriculum that thinks it teaches it.
    #[test]
    fn every_lesson_actually_generates_on_the_inference_canvas() {
        for &kind in LessonKind::all() {
            let built = (0..50u64)
                .filter(|&seed| generate(kind, INFERENCE, seed).is_some())
                .count();
            assert!(
                built > 0,
                "{} never generated on {INFERENCE} in 50 seeds",
                kind.name()
            );
        }
    }

    /// The default pool is a pool of *shapes*, not of sizes: it has to contain
    /// non-squares, or the curriculum is the square one under a longer name.
    #[test]
    fn the_default_pool_is_mostly_not_square() {
        let pool = Canvas::pool(DEFAULT_CANVAS_MIN, DEFAULT_CANVAS_MAX);
        let squares = pool.iter().filter(|c| c.width == c.height).count();
        assert!(
            pool.len() > 2 * squares,
            "{squares} of {} canvases are square -- the pool barely varies shape",
            pool.len()
        );
        assert!(
            pool.contains(&INFERENCE),
            "the pool does not contain {INFERENCE}, the shape the issue infers on"
        );
    }
}
