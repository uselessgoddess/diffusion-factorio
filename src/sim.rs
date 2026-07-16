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

use crate::world::{Entity, Grid, Item, Misc};
use std::collections::VecDeque;

/// Maximum tiles an underground belt can span (yellow-belt reach), mirroring
/// `UNDERGROUND_REACH` in the reference.
pub const UNDERGROUND_REACH: i32 = 5;

/// Every inserter standing beside `(x, y)` whose hand reaches **into** it.
///
/// An inserter's pickup is the tile behind it (`q - d`), the same rule
/// `throughput::accepts_from` applies from the other side. Facing matters: an
/// inserter *pointing at* the belt is dropping onto it, not taking off it, and
/// is not a tap.
fn tapping_inserters(grid: &Grid, x: usize, y: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if !grid.in_bounds(nx, ny) {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        let neighbour = grid.get(nx, ny);
        if neighbour.entity != Entity::Inserter {
            continue;
        }
        let (idx, idy) = neighbour.direction.delta();
        if (nx as i32 - idx, ny as i32 - idy) == (x as i32, y as i32) && !out.contains(&(nx, ny)) {
            out.push((nx, ny));
        }
    }
    out
}

/// Every entity this one pushes flow *into*, named by its anchor. A belt at `p`
/// facing `d` pushes flow to `p + d` **and** to any inserter reaching in to take
/// off it; underground entrances jump to the nearest matching exit within reach;
/// a source or an assembler offers to everything around its footprint.
///
/// `(x, y)` must be an anchor — the results are anchors too, so callers compose
/// without ever handling a body tile. See [`Grid::anchor_at`].
///
/// This says only who *offers* to whom. Whether the receiver takes it is the
/// receiver's business — see `throughput::accepts_from`. [`item_reaches_sink`]
/// is deliberately laxer and accepts from any pusher.
///
/// **The sideways offer is the bus tap**, and it used to be missing. A belt
/// offered flow only to the one tile it faced, so an inserter standing beside a
/// passing line was never offered anything and the oldest layout in Factorio —
/// one belt, a row of inserters pulling off it into a row of machines — routed
/// nothing at all (`experiments/bus_tap`: score 0.000, does not even reach a
/// sink). The pattern was unrepresentable, so no lesson could teach it and
/// Best-of-N would have thrown it away on sight.
///
/// Scope, deliberately: a plain [`Entity::TransportBelt`] is tappable and an
/// underground belt or a splitter is not. In the game all three are, but the
/// issue's "одна линия" is a belt, and the other two would each need their own
/// evidence rather than an assumption (`docs/ROADMAP.md`).
pub(crate) fn flow_targets(grid: &Grid, x: usize, y: usize) -> Vec<(usize, usize)> {
    let cell = grid.get(x, y);
    let mut out = Vec::new();
    match cell.entity {
        Entity::TransportBelt | Entity::Inserter => {
            let (dx, dy) = cell.direction.delta();
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if grid.in_bounds(nx, ny) {
                out.push((nx as usize, ny as usize));
            }
            // An inserter takes off the tile behind it; it does not offer to
            // whatever is beside it.
            if cell.entity == Entity::TransportBelt {
                for tap in tapping_inserters(grid, x, y) {
                    if !out.contains(&tap) {
                        out.push(tap);
                    }
                }
            }
        }
        Entity::Splitter => {
            // A splitter is 2×1, and that is the whole entity: each of its two
            // tiles pushes forward independently, which is how one belt becomes
            // two. Modelled as 1×1 it had a single output and was an ordinary
            // belt wearing a different glyph.
            let (dx, dy) = cell.direction.delta();
            for (tx, ty) in grid.footprint_at(x, y) {
                let (nx, ny) = (tx + dx, ty + dy);
                if grid.in_bounds(nx, ny) {
                    out.push((nx as usize, ny as usize));
                }
            }
        }
        Entity::Assembler | Entity::Source => {
            // A machine consumes an input and makes its product available to any
            // pickup standing on its perimeter; a source anchor offers its item
            // the same way. Re-emit to every loading slot; the BFS `visited`
            // guard and the inserter's own facing decide where the flow can
            // actually continue.
            out.extend(grid.perimeter(x, y));
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
    resolve(grid, (x, y), out)
}

/// Name each tile by the entity standing on it, drop the ones standing on
/// nothing (and the offering entity itself), and deduplicate.
///
/// A 3×3 machine covers nine tiles but is *one* node. Without the dedupe an
/// assembler whose neighbour is another assembler would appear to have three
/// successors where it has one, and `throughput`'s even fan-out split would
/// quietly divide its output by three.
fn resolve(grid: &Grid, from: (usize, usize), tiles: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(tiles.len());
    for (tx, ty) in tiles {
        match grid.anchor_at(tx, ty) {
            Some(anchor) if anchor != from && !out.contains(&anchor) => out.push(anchor),
            _ => {}
        }
    }
    out
}

/// Does a source's item actually reach a sink that accepts it?
///
/// A source feeds any orthogonally adjacent belt; flow then follows belt
/// directions until it reaches (points into) a sink or dead-ends. The carried
/// item travels with the flow: belts pass it through unchanged, an assembler
/// consumes its recipe's ingredients and emits the product, and a sink only
/// accepts the item it is configured for.
///
/// Tracking the item is what stops the check rewarding a factory that merely
/// *connects* source to sink. Without it, belting raw iron plates straight into
/// a gear sink scores the same as building the gear assembler — the reference
/// guards the same hole in `factorion_rs/src/throughput.rs:205-226`, where
/// sinks only score their configured item.
///
/// This is a fixpoint over *sets*, not a walk carrying one item, and it has to
/// be. An electronic circuit needs an iron plate **and** copper cable, so
/// whether its machine runs is a fact about everything that has ever arrived
/// there — not about the item currently in hand. A walk reaches the machine
/// holding iron, asks "can I craft?", and cannot answer, because the cable's
/// arrival is a different branch of the same search. Arriving-item sets only
/// grow, so iterating to a fixpoint terminates.
pub fn item_reaches_sink(grid: &Grid) -> bool {
    let sources = positions(grid, Entity::Source);
    let sinks = positions(grid, Entity::Sink);
    if sources.is_empty() || sinks.is_empty() {
        return false;
    }

    // `arriving[cell]` = every item that can be delivered into that cell.
    let mut arriving = vec![[false; Item::COUNT]; grid.len()];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    // Seed: whatever a source offers to is an entry point, and what shows up
    // there is that source's item.
    for &(sx, sy) in &sources {
        let carried = grid.get(sx, sy).item;
        for (nx, ny) in flow_targets(grid, sx, sy) {
            if is_conveyor(grid, nx, ny) {
                deliver(grid, &mut arriving, &mut queue, (nx, ny), carried);
            }
        }
    }

    while let Some((x, y)) = queue.pop_front() {
        // What this cell can hand on, given everything that has reached it.
        let offered = emits(grid, &arriving[grid.idx(x, y)], x, y);
        for carried in offered {
            for (tx, ty) in flow_targets(grid, x, y) {
                if sinks.contains(&(tx, ty)) && sink_accepts(grid.get(tx, ty).item, carried) {
                    return true;
                }
                if is_conveyor(grid, tx, ty) {
                    deliver(grid, &mut arriving, &mut queue, (tx, ty), carried);
                }
            }
        }
    }
    false
}

/// What a cell offers onward, given everything that has arrived at it.
///
/// A conveyor passes on whatever it was handed. An assembler is the interesting
/// case: it emits its product, and only its product, and only once **every**
/// ingredient has turned up.
fn emits(grid: &Grid, arriving: &[bool; Item::COUNT], x: usize, y: usize) -> Vec<Item> {
    let cell = grid.get(x, y);
    if cell.entity != Entity::Assembler {
        return (0..Item::COUNT)
            .filter(|&i| arriving[i])
            .filter_map(Item::from_id)
            .collect();
    }
    // An untagged assembler has no recipe, so it crafts nothing.
    let Some(recipe) = cell.item.recipe() else {
        return Vec::new();
    };
    let fed = recipe.ingredients.iter().all(|i| arriving[i.item as usize]);
    if fed {
        vec![cell.item]
    } else {
        Vec::new()
    }
}

/// An untagged sink has no filter and accepts anything; a tagged one accepts
/// only its own item.
pub(crate) fn sink_accepts(filter: Item, carried: Item) -> bool {
    filter == Item::None || filter == carried
}

/// Record that `carried` reaches `(x, y)`, and re-wake the cell if that is news.
///
/// Re-queueing on *any* new arrival is what makes the fixpoint a fixpoint: an
/// assembler visited while only iron had arrived emitted nothing, and must be
/// asked again once the cable turns up. Each cell can only be woken
/// [`Item::COUNT`] times, because the flags never go back to `false`.
fn deliver(
    grid: &Grid,
    arriving: &mut [[bool; Item::COUNT]],
    queue: &mut VecDeque<(usize, usize)>,
    (x, y): (usize, usize),
    carried: Item,
) {
    let seen = &mut arriving[grid.idx(x, y)][carried as usize];
    if !*seen {
        *seen = true;
        queue.push_back((x, y));
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

    /// A crafting line, with the assembler occupying the nine tiles it occupies
    /// in Factorio:
    ///
    /// ```text
    ///     . . A A A . .
    ///     S i A A A i K
    ///     . . A A A . .
    /// ```
    ///
    /// Only the anchor at (2, 0) holds the assembler; the other eight tiles read
    /// as `Empty` and resolve back to it through [`Grid::anchor_at`]. The
    /// inserters stand on the machine's perimeter rather than beside a 1×1 stand-in,
    /// which is what a real `assembling-machine-1` demands.
    fn crafting_line(source_item: Item) -> Grid {
        let mut g = Grid::new(7, 3);
        g.set(
            0,
            1,
            Cell {
                entity: Entity::Source,
                item: source_item,
                ..Default::default()
            },
        );
        for x in [1, 5] {
            g.set(
                x,
                1,
                Cell {
                    entity: Entity::Inserter,
                    direction: Direction::East,
                    ..Default::default()
                },
            );
        }
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
            6,
            1,
            Cell {
                entity: Entity::Sink,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        g
    }

    /// The same layout with the assembler actually built does work.
    #[test]
    fn assembler_crafting_the_sink_item_is_functional() {
        let g = crafting_line(Item::IronPlate);
        assert!(
            g.is_consistent(),
            "the line must be buildable to begin with"
        );
        assert!(item_reaches_sink(&g));
    }

    /// An assembler fed the wrong ingredient never crafts.
    #[test]
    fn assembler_fed_wrong_ingredient_is_not_functional() {
        // Gears need iron.
        assert!(!item_reaches_sink(&crafting_line(Item::CopperPlate)));
    }

    /// An inserter loading a machine through one of its *body* tiles feeds the
    /// machine, exactly as it does in Factorio:
    ///
    /// ```text
    ///     . . . S . . .
    ///     . . . i . . .   <- points south, into a tile that reads as `Empty`
    ///     . . A A A . .
    ///     . . A A A i K
    ///     . . A A A . .
    /// ```
    ///
    /// The tile at (3, 1) hands off to (3, 2), which stores nothing: the
    /// assembler lives at its anchor (2, 2). Were the sim to read the target
    /// tile directly it would see an empty cell and drop the flow, and only
    /// inserters that happened to face the anchor's own tile would work.
    #[test]
    fn a_machine_is_loaded_through_a_body_tile() {
        let mut g = Grid::new(7, 5);
        g.set(
            3,
            0,
            Cell {
                entity: Entity::Source,
                item: Item::IronPlate,
                ..Default::default()
            },
        );
        g.set(
            3,
            1,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::South,
                ..Default::default()
            },
        );
        g.set(
            2,
            2,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        g.set(
            5,
            3,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(
            6,
            3,
            Cell {
                entity: Entity::Sink,
                item: Item::IronGear,
                ..Default::default()
            },
        );

        assert_eq!(g.anchor_at(3, 2), Some((2, 2)), "the loaded tile is a body");
        assert!(g.get(3, 2).is_empty(), "and it stores nothing");
        assert!(item_reaches_sink(&g));
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
