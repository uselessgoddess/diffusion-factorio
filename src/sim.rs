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

use crate::world::{Direction, Entity, Grid, Item, Misc};
use std::collections::VecDeque;

/// Maximum tiles an underground belt can span (yellow-belt reach), mirroring
/// `UNDERGROUND_REACH` in the reference.
pub const UNDERGROUND_REACH: i32 = 5;

/// Every cell this one pushes flow *into*. A belt at `p` facing `d` pushes flow
/// to `p + d`; underground entrances jump to the nearest matching exit within
/// reach; a source or an assembler offers to every orthogonal neighbour.
///
/// This says only who *offers* to whom. Whether the receiver takes it is the
/// receiver's business — see `throughput::accepts_from`. [`item_reaches_sink`]
/// is deliberately laxer and accepts from any pusher.
pub(crate) fn flow_targets(grid: &Grid, x: usize, y: usize) -> Vec<(usize, usize)> {
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
        Entity::Assembler | Entity::Source => {
            // A machine consumes an input and makes its product available to any
            // adjacent pickup (inserter); a source anchor offers its item the
            // same way. Re-emit to every orthogonal neighbour; the BFS `visited`
            // guard and the inserter's own facing decide where the flow can
            // actually continue.
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

/// Does a source's item actually reach a sink that accepts it?
///
/// A source feeds any orthogonally adjacent belt; flow then follows belt
/// directions until it reaches (points into) a sink or dead-ends. The carried
/// item travels with the flow: belts pass it through unchanged, an assembler
/// consumes its recipe's ingredient and emits the product, and a sink only
/// accepts the item it is configured for.
///
/// Tracking the item is what stops the check rewarding a factory that merely
/// *connects* source to sink. Without it, belting raw iron plates straight into
/// a gear sink scores the same as building the gear assembler — the reference
/// guards the same hole in `factorion_rs/src/throughput.rs:205-226`, where
/// sinks only score their configured item.
pub fn item_reaches_sink(grid: &Grid) -> bool {
    let sources = positions(grid, Entity::Source);
    let sinks = positions(grid, Entity::Sink);
    if sources.is_empty() || sinks.is_empty() {
        return false;
    }

    // State is (cell, carried item): the same tile can legitimately be reached
    // carrying different items, and only some of them satisfy the sink.
    let mut visited = vec![[false; Item::COUNT]; grid.len()];
    let mut queue: VecDeque<((usize, usize), Item)> = VecDeque::new();

    // Seed: belts orthogonally adjacent to any source are entry points, and
    // they start out carrying that source's item.
    for &(sx, sy) in &sources {
        let carried = grid.get(sx, sy).item;
        for (nx, ny) in orthogonal(grid, sx, sy) {
            if is_conveyor(grid, nx, ny) {
                push(grid, &mut visited, &mut queue, (nx, ny), carried);
            }
        }
    }

    while let Some(((x, y), carried)) = queue.pop_front() {
        // An assembler transforms what passes through it: it only runs if fed
        // its ingredient, and what leaves is the product.
        let carried = match transform(grid, x, y, carried) {
            Some(item) => item,
            None => continue, // wrong ingredient: the machine never crafts
        };

        for (tx, ty) in flow_targets(grid, x, y) {
            if sinks.contains(&(tx, ty)) && sink_accepts(grid.get(tx, ty).item, carried) {
                return true;
            }
            if is_conveyor(grid, tx, ty) {
                push(grid, &mut visited, &mut queue, (tx, ty), carried);
            }
        }
    }
    false
}

/// What leaves a cell given what entered it. `None` means the flow stops here.
fn transform(grid: &Grid, x: usize, y: usize, carried: Item) -> Option<Item> {
    let cell = grid.get(x, y);
    if cell.entity != Entity::Assembler {
        return Some(carried);
    }
    // An untagged assembler has no recipe, so it crafts nothing.
    let ingredient = cell.item.ingredient()?;
    (ingredient == carried).then_some(cell.item)
}

/// An untagged sink has no filter and accepts anything; a tagged one accepts
/// only its own item.
pub(crate) fn sink_accepts(filter: Item, carried: Item) -> bool {
    filter == Item::None || filter == carried
}

fn push(
    grid: &Grid,
    visited: &mut [[bool; Item::COUNT]],
    queue: &mut VecDeque<((usize, usize), Item)>,
    (x, y): (usize, usize),
    carried: Item,
) {
    let seen = &mut visited[grid.idx(x, y)][carried as usize];
    if !*seen {
        *seen = true;
        queue.push_back(((x, y), carried));
    }
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

    /// Belt a raw plate straight into a sink that wants a crafted item and the
    /// factory is *not* functional, however well-connected it looks. This is
    /// the reward hack the check exists to reject.
    #[test]
    fn raw_input_belted_to_a_crafted_sink_is_not_functional() {
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
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, Cell::belt(Direction::East));
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        assert!(!item_reaches_sink(&g));
    }

    /// The same layout with the assembler actually built does work.
    #[test]
    fn assembler_crafting_the_sink_item_is_functional() {
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
        g.set(
            1,
            0,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(
            2,
            0,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        g.set(
            3,
            0,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        assert!(item_reaches_sink(&g));
    }

    /// An assembler fed the wrong ingredient never crafts.
    #[test]
    fn assembler_fed_wrong_ingredient_is_not_functional() {
        let mut g = Grid::new(5, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                item: Item::CopperPlate, // gears need iron
                ..Default::default()
            },
        );
        g.set(
            1,
            0,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(
            2,
            0,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        g.set(
            3,
            0,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        assert!(!item_reaches_sink(&g));
    }

    /// Delivering the wrong plate to a plate sink is still wrong.
    #[test]
    fn mismatched_raw_items_do_not_connect() {
        let mut g = Grid::new(4, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                item: Item::IronPlate,
                ..Default::default()
            },
        );
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, Cell::belt(Direction::East));
        g.set(
            3,
            0,
            Cell {
                entity: Entity::Sink,
                item: Item::CopperPlate,
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
