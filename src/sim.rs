//! Lightweight functional evaluation of a factory grid.
//!
//! The reference project computes throughput by building a lane-aware flow
//! graph and running a topological flow propagation with cycle detection
//! (`factorion_rs/src/{graph,throughput}.rs`). That is powerful but heavy; full
//! parity is tracked as a roadmap item (`docs/ROADMAP.md`).
//!
//! For validating that the diffusion model learns something *functional* (not
//! just per-token accuracy), we implement a cheaper but meaningful check for the
//! belt-routing lessons: does the item actually flow from every source to a
//! sink along the placed belts? This is the analogue of the reference's
//! "normalized throughput" reward — a task-level, simulator-grounded signal that
//! per-cell accuracy alone cannot capture.

use crate::world::{Direction, Entity, Grid, Misc};
use std::collections::VecDeque;

/// Maximum tiles an underground belt can span (yellow-belt reach), mirroring
/// `UNDERGROUND_REACH` in the reference.
pub const UNDERGROUND_REACH: i32 = 5;

/// Follow the belt network from a starting belt cell; return every belt/exit
/// cell the flow reaches. A belt at `p` facing `d` pushes flow to `p + d`.
/// Underground entrances jump to the nearest matching exit within reach.
fn flow_targets(grid: &Grid, x: usize, y: usize) -> Vec<(usize, usize)> {
    let cell = grid.get(x, y);
    let mut out = Vec::new();
    match cell.entity {
        Entity::TransportBelt | Entity::Splitter | Entity::Inserter => {
            let (dx, dy) = cell.direction.delta();
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if grid.in_bounds(nx, ny) {
                out.push((nx as usize, ny as usize));
            }
        }
        Entity::Assembler => {
            // A machine consumes an input and makes its product available to any
            // adjacent pickup (inserter). Re-emit to every orthogonal neighbour;
            // the BFS `visited` guard and the inserter's own facing decide where
            // the flow can actually continue.
            out.extend(orthogonal(grid, x, y));
        }
        Entity::UndergroundBelt => {
            let (dx, dy) = cell.direction.delta();
            if cell.misc == Misc::UndergroundUp {
                // Exit: behaves like a belt, pushes flow forward one tile.
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if grid.in_bounds(nx, ny) {
                    out.push((nx as usize, ny as usize));
                }
            } else {
                // Entrance: tunnel to the nearest matching exit along `d`.
                for step in 1..=UNDERGROUND_REACH {
                    let (nx, ny) = (x as i32 + dx * step, y as i32 + dy * step);
                    if !grid.in_bounds(nx, ny) {
                        break;
                    }
                    let c = grid.get(nx as usize, ny as usize);
                    if c.entity == Entity::UndergroundBelt
                        && c.misc == Misc::UndergroundUp
                        && c.direction == cell.direction
                    {
                        out.push((nx as usize, ny as usize));
                        break;
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// Does the item reach a sink from at least one source, following belts?
///
/// A source feeds any orthogonally adjacent belt; flow then follows belt
/// directions until it reaches (points into) a sink or dead-ends.
pub fn item_reaches_sink(grid: &Grid) -> bool {
    let sources: Vec<(usize, usize)> = positions(grid, Entity::Source);
    let sinks: Vec<(usize, usize)> = positions(grid, Entity::Sink);
    if sources.is_empty() || sinks.is_empty() {
        return false;
    }

    let mut visited = vec![false; grid.len()];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    // Seed: belts orthogonally adjacent to any source are entry points.
    for &(sx, sy) in &sources {
        for (nx, ny) in orthogonal(grid, sx, sy) {
            if is_conveyor(grid, nx, ny) && !visited[grid.idx(nx, ny)] {
                visited[grid.idx(nx, ny)] = true;
                queue.push_back((nx, ny));
            }
        }
    }

    while let Some((x, y)) = queue.pop_front() {
        for (tx, ty) in flow_targets(grid, x, y) {
            if sinks.contains(&(tx, ty)) {
                return true;
            }
            if is_conveyor(grid, tx, ty) && !visited[grid.idx(tx, ty)] {
                visited[grid.idx(tx, ty)] = true;
                queue.push_back((tx, ty));
            }
        }
    }
    false
}

fn is_conveyor(grid: &Grid, x: usize, y: usize) -> bool {
    matches!(
        grid.get(x, y).entity,
        Entity::TransportBelt
            | Entity::UndergroundBelt
            | Entity::Splitter
            | Entity::Inserter
            | Entity::Assembler
    )
}

fn positions(grid: &Grid, kind: Entity) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if grid.get(x, y).entity == kind {
                v.push((x, y));
            }
        }
    }
    v
}

fn orthogonal(grid: &Grid, x: usize, y: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for d in [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ] {
        let (dx, dy) = d.delta();
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if grid.in_bounds(nx, ny) {
            v.push((nx as usize, ny as usize));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Cell, Direction, Entity};

    #[test]
    fn straight_belt_connects() {
        let mut g = Grid::new(5, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                ..Default::default()
            },
        );
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, Cell::belt(Direction::East));
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                ..Default::default()
            },
        );
        assert!(item_reaches_sink(&g));
    }

    #[test]
    fn broken_belt_does_not_connect() {
        let mut g = Grid::new(5, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                ..Default::default()
            },
        );
        g.set(1, 0, Cell::belt(Direction::East));
        // gap at x=2
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                ..Default::default()
            },
        );
        assert!(!item_reaches_sink(&g));
    }

    #[test]
    fn wrong_direction_does_not_connect() {
        let mut g = Grid::new(5, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                ..Default::default()
            },
        );
        g.set(1, 0, Cell::belt(Direction::West)); // points back at source
        g.set(2, 0, Cell::belt(Direction::East));
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                ..Default::default()
            },
        );
        assert!(!item_reaches_sink(&g));
    }

    #[test]
    fn underground_tunnels_across_gap() {
        let mut g = Grid::new(6, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                ..Default::default()
            },
        );
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(
            2,
            0,
            Cell {
                entity: Entity::UndergroundBelt,
                direction: Direction::East,
                misc: Misc::UndergroundDown,
                ..Default::default()
            },
        );
        g.set(
            4,
            0,
            Cell {
                entity: Entity::UndergroundBelt,
                direction: Direction::East,
                misc: Misc::UndergroundUp,
                ..Default::default()
            },
        );
        g.set(
            5,
            0,
            Cell {
                entity: Entity::Sink,
                ..Default::default()
            },
        );
        assert!(item_reaches_sink(&g));
    }
}
