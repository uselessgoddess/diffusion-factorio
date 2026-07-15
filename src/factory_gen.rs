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
}

impl LessonKind {
    pub fn all() -> &'static [LessonKind] {
        &[
            LessonKind::MoveOneItem,
            LessonKind::MoveOneItemChaos,
            LessonKind::AssemblerLine,
            LessonKind::UndergroundCross,
        ]
    }
    pub fn name(self) -> &'static str {
        match self {
            LessonKind::MoveOneItem => "MOVE_ONE_ITEM",
            LessonKind::MoveOneItemChaos => "MOVE_ONE_ITEM_CHAOS",
            LessonKind::AssemblerLine => "ASSEMBLER_LINE",
            LessonKind::UndergroundCross => "UNDERGROUND_CROSS",
        }
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
}
