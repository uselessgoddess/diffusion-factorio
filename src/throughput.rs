//! Graded throughput: *how well* does a factory work, not just whether it does.
//!
//! [`sim::item_reaches_sink`](crate::sim::item_reaches_sink) answers a yes/no
//! question, and that is the project's central bottleneck (`docs/ROADMAP.md`):
//! a binary metric cannot rank two working factories, so Best-of-N has nothing
//! to sort by and RL has no gradient to climb — its reward is already saturated
//! at 1.0. This module supplies the ranking signal: a factory's items/second.
//!
//! The design is ported from the reference project's
//! `factorion_rs/src/{graph,throughput}.rs`, with three deliberate departures,
//! each of which is a test in this file:
//!
//! * **The assembler is a real machine.** The reference never reads
//!   `crafting_time` or `crafting_speed` (verified: they appear nowhere in its
//!   `entities.rs`/`throughput.rs`). It models a machine as a pass-through
//!   *ratio* capped at 1.0, which silently reinterprets a per-craft count as a
//!   per-second rate. That happens to be right for 0.5 s recipes and is 12–20×
//!   too generous for the long ones, and it means a machine can never be a
//!   bottleneck — which is exactly what a machine usually *is*. We cap output at
//!   the machine's real rate ([`Recipe::crafts_per_second`]).
//! * **Cycles degrade locally.** The reference scores the *whole* factory 0 if a
//!   cycle exists anywhere, even in a disconnected corner. As a training signal
//!   that is a cliff. Here a cycle simply never gets a topological turn, so it
//!   delivers nothing and starves whatever is downstream of it; sinks fed by
//!   other paths still score. This needs no cycle check at all — it is what
//!   Kahn's algorithm already does.
//! * **No lanes.** The reference splits every belt tile into a left/right lane
//!   node to model sideloading. Our world model has no lanes and 1×1 entities,
//!   so lane-awareness is vacuous here: an inserter has exactly one pickup tile,
//!   and belt merging is handled by the per-tile cap. This is a limitation of
//!   the world model, not of the port (`docs/ROADMAP.md`).
//!
//! Rates are vanilla items/second. The score is a **per-sink mean, not a sum**:
//! two sinks fed 15/s each score 15.0, not 30.0.

use crate::sim::{flow_targets, sink_accepts};
use crate::world::{Entity, Grid, Item};
use std::collections::VecDeque;

/// Exponent of the power mean used to fold per-sink rates into one score.
///
/// At `p = 0.5` the mean sits between the arithmetic (`p = 1`, which lets one
/// saturated sink hide a dead one) and the geometric (`p → 0`, which zeroes the
/// whole factory if any single sink is starved). It penalises imbalance without
/// the cliff: feeding one of two sinks at 15/s scores 3.75, not 7.5 and not 0.
pub const FACTORY_SCORE_P: f64 = 0.5;

/// Items/second one transport-belt tile carries (vanilla yellow belt).
pub const BELT_RATE: f64 = 15.0;

/// Items/second one inserter moves (vanilla, un-researched).
pub const INSERTER_RATE: f64 = 0.86;

/// Flow through a node, in items/second, indexed by `Item as usize`.
type Flow = [f64; Item::COUNT];

const NO_FLOW: Flow = [0.0; Item::COUNT];

/// What one sink actually receives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkDelivery {
    pub at: (usize, usize),
    /// The item this sink is configured to accept; `Item::None` means unfiltered.
    pub item: Item,
    /// Items/second of accepted items arriving.
    pub achieved: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThroughputReport {
    pub deliveries: Vec<SinkDelivery>,
    /// Power mean of the deliveries at [`FACTORY_SCORE_P`], in items/second.
    pub score: f64,
}

/// Generalised mean `((1/N)·Σ vᵢ^p)^(1/p)`. Empty input scores 0.
pub fn power_mean(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|v| v.powf(p)).sum();
    (sum / values.len() as f64).powf(1.0 / p)
}

/// Fold per-sink rates into a single score.
///
/// Every sink is in the denominator whether or not it is fed, so an ignored
/// sink drags the score down rather than being quietly dropped. A non-finite
/// mean (a source feeding a sink directly, with nothing built in between)
/// collapses to 0 rather than winning.
pub fn factory_score(deliveries: &[SinkDelivery]) -> f64 {
    let achieved: Vec<f64> = deliveries.iter().map(|d| d.achieved).collect();
    let score = power_mean(&achieved, FACTORY_SCORE_P);
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

/// Items/second this factory delivers. See [`throughput`] for the per-sink
/// breakdown.
pub fn score(grid: &Grid) -> f64 {
    throughput(grid).score
}

/// Propagate flow from every source and report what each sink receives.
pub fn throughput(grid: &Grid) -> ThroughputReport {
    let (successors, predecessors) = build_graph(grid);
    let n = grid.len();

    let sources = positions(grid, Entity::Source);
    let sinks = positions(grid, Entity::Sink);
    if sources.is_empty() || sinks.is_empty() {
        return ThroughputReport {
            deliveries: Vec::new(),
            score: 0.0,
        };
    }

    // Only nodes a source can actually reach take part. Counting unreachable
    // predecessors in the in-degree would deadlock the queue behind an orphan
    // feeder that never fires — a belt someone laid pointing at ours from
    // nowhere would stall the whole factory.
    let reachable = reachable_from(&successors, &sources);
    let mut in_degree: Vec<usize> = (0..n)
        .map(|i| predecessors[i].iter().filter(|&&p| reachable[p]).count())
        .collect();

    let mut outputs: Vec<Flow> = vec![NO_FLOW; n];
    let mut done = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &s in &sources {
        // A source is an unlimited supply of its item. This seeding is the only
        // place flow enters the factory.
        outputs[s][grid.cells[s].item as usize] = f64::INFINITY;
        queue.push_back(s);
    }

    // Kahn's algorithm: a node fires once every reachable predecessor has. Any
    // cycle never reaches in-degree 0, so it silently delivers nothing.
    while let Some(u) = queue.pop_front() {
        if done[u] {
            continue;
        }
        done[u] = true;

        // Sources ignore their input and keep the seeded infinity.
        if grid.cells[u].entity != Entity::Source {
            let mut input = NO_FLOW;
            for &p in &predecessors[u] {
                for (acc, rate) in input.iter_mut().zip(outputs[p].iter()) {
                    *acc += rate;
                }
            }
            outputs[u] = transform(grid, u, &input);

            // Fan-out is an even split: what leaves this tile is shared between
            // everything it feeds. This is the only splitting mechanism, and it
            // has no backpressure — a machine feeding one live inserter and one
            // dead-ended inserter still sends half its output to the dead end.
            let k = successors[u].len();
            if k > 1 {
                for rate in outputs[u].iter_mut() {
                    *rate /= k as f64;
                }
            }
        }

        for &v in &successors[u] {
            in_degree[v] = in_degree[v].saturating_sub(1);
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    let deliveries: Vec<SinkDelivery> = sinks
        .iter()
        .map(|&s| {
            let filter = grid.cells[s].item;
            // Only what the sink is configured for counts. Without this a policy
            // could belt raw plates into a gear sink and score full marks for
            // never building the machine.
            let achieved = (0..Item::COUNT)
                .filter(|&i| sink_accepts(filter, Item::from_id(i).expect("in range")))
                .map(|i| outputs[s][i])
                .sum();
            SinkDelivery {
                at: (s % grid.width, s / grid.width),
                item: filter,
                achieved,
            }
        })
        .collect();

    let score = factory_score(&deliveries);
    ThroughputReport { deliveries, score }
}

/// Adjacency over cells: an edge `p -> q` exists iff `p` pushes into `q` *and*
/// `q` takes from `p`.
fn build_graph(grid: &Grid) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = grid.len();
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for y in 0..grid.height {
        for x in 0..grid.width {
            let p = grid.idx(x, y);
            for (tx, ty) in flow_targets(grid, x, y) {
                if !accepts_from(grid, (tx, ty), (x, y)) {
                    continue;
                }
                let q = grid.idx(tx, ty);
                if !successors[p].contains(&q) {
                    successors[p].push(q);
                    predecessors[q].push(p);
                }
            }
        }
    }
    (successors, predecessors)
}

/// Will the cell at `q` take flow offered by the cell at `p`?
///
/// [`flow_targets`] says who offers; this says who takes. Splitting the two is
/// what keeps an inserter from magically grabbing off a belt beside it.
fn accepts_from(grid: &Grid, (qx, qy): (usize, usize), (px, py): (usize, usize)) -> bool {
    let q = grid.get(qx, qy);
    match q.entity {
        Entity::Sink => true,
        // An inserter's hand reaches exactly one tile: the one behind it.
        Entity::Inserter => {
            let (dx, dy) = q.direction.delta();
            (px as i32, py as i32) == (qx as i32 - dx, qy as i32 - dy)
        }
        // A machine is loaded by an inserter swinging into it — a belt running
        // past its wall delivers nothing. A source anchor stands in for an
        // inserter so a lesson can feed a machine directly.
        Entity::Assembler => matches!(grid.get(px, py).entity, Entity::Inserter | Entity::Source),
        // Conveyors take from anything feeding in, except the tile they
        // themselves feed: two belts nose to nose deadlock, they do not loop.
        Entity::TransportBelt | Entity::UndergroundBelt | Entity::Splitter => {
            !flow_targets(grid, qx, qy).contains(&(px, py))
        }
        Entity::Empty | Entity::Source => false,
    }
}

/// What leaves a node given what entered it.
fn transform(grid: &Grid, idx: usize, input: &Flow) -> Flow {
    let cell = grid.cells[idx];
    match cell.entity {
        // The machine model. Unlike the reference we cap at the machine's own
        // rate, so over-feeding an assembler is worth nothing and the machine
        // can be the bottleneck.
        Entity::Assembler => {
            let mut out = NO_FLOW;
            if let Some(recipe) = cell.item.recipe() {
                let supplied = input[recipe.ingredient as usize] / recipe.ingredient_qty;
                let crafts = supplied.min(recipe.crafts_per_second());
                out[cell.item as usize] = crafts * recipe.output_qty;
            }
            out
        }
        Entity::TransportBelt | Entity::UndergroundBelt | Entity::Splitter => {
            clamp_total(input, BELT_RATE)
        }
        Entity::Inserter => clamp_total(input, INSERTER_RATE),
        Entity::Sink => *input,
        Entity::Source | Entity::Empty => NO_FLOW,
    }
}

/// Cap a tile's *total* throughput, scaling each item's share to fit.
///
/// The reference caps each item separately, which lets one tile carry a full
/// belt of every item at once. A tile moves so many items per second regardless
/// of what they are.
fn clamp_total(flow: &Flow, cap: f64) -> Flow {
    let total: f64 = flow.iter().sum();
    if total <= cap {
        return *flow;
    }
    let mut out = NO_FLOW;
    let unlimited = flow.iter().filter(|r| r.is_infinite()).count();
    if unlimited > 0 {
        // An unlimited supply crowds every finite one off the tile, and shares
        // it evenly with any other unlimited supply feeding in.
        let each = cap / unlimited as f64;
        for (o, r) in out.iter_mut().zip(flow.iter()) {
            if r.is_infinite() {
                *o = each;
            }
        }
    } else {
        let scale = cap / total;
        for (o, r) in out.iter_mut().zip(flow.iter()) {
            *o = r * scale;
        }
    }
    out
}

fn reachable_from(successors: &[Vec<usize>], roots: &[usize]) -> Vec<bool> {
    let mut seen = vec![false; successors.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &r in roots {
        seen[r] = true;
        queue.push_back(r);
    }
    while let Some(u) = queue.pop_front() {
        for &v in &successors[u] {
            if !seen[v] {
                seen[v] = true;
                queue.push_back(v);
            }
        }
    }
    seen
}

fn positions(grid: &Grid, kind: Entity) -> Vec<usize> {
    (0..grid.len())
        .filter(|&i| grid.cells[i].entity == kind)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory_gen::{self, LessonKind};
    use crate::sim::item_reaches_sink;
    use crate::world::{Cell, Direction, Misc};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

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

    /// `S i a i K`: the assembler_line lesson, which the model already solves
    /// with exact=0.99 — and which the binary metric scores 1.0 either way.
    fn assembler_line(recipe: Item, ingredient: Item) -> Grid {
        let mut g = Grid::new(5, 1);
        g.set(0, 0, anchor(Entity::Source, ingredient));
        g.set(1, 0, inserter(Direction::East));
        g.set(2, 0, assembler(recipe));
        g.set(3, 0, inserter(Direction::East));
        g.set(4, 0, anchor(Entity::Sink, recipe));
        g
    }

    #[test]
    fn power_mean_matches_the_arithmetic_mean_at_p_one() {
        assert!(approx(power_mean(&[1.0, 3.0], 1.0), 2.0));
        assert!(approx(power_mean(&[4.0, 4.0], 0.5), 4.0));
        assert_eq!(power_mean(&[], 0.5), 0.0);
    }

    #[test]
    fn a_belt_run_delivers_a_full_belt() {
        let mut g = Grid::new(5, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, Cell::belt(Direction::East));
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(4, 0, anchor(Entity::Sink, Item::IronPlate));
        assert!(approx(score(&g), BELT_RATE));
    }

    #[test]
    fn a_gap_in_the_belt_delivers_nothing() {
        let mut g = Grid::new(5, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        // gap at x=2
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(4, 0, anchor(Entity::Sink, Item::IronPlate));
        assert_eq!(score(&g), 0.0);
    }

    /// The whole point. Two factories that the binary metric calls identical —
    /// both `functional` — deliver measurably different rates, because a gear
    /// costs 2 plates a craft and a cable yields 2 items a craft.
    #[test]
    fn the_recipe_decides_the_rate_of_an_identical_layout() {
        let gears = assembler_line(Item::IronGear, Item::IronPlate);
        let cable = assembler_line(Item::CopperCable, Item::CopperPlate);
        assert!(item_reaches_sink(&gears) && item_reaches_sink(&cable));

        // The input inserter admits 0.86 plate/s. Gears burn 2 plates a craft,
        // so 0.43 craft/s -> 0.43 gear/s, and the output inserter is not even
        // the constraint.
        assert!(approx(score(&gears), 0.43), "gears: {}", score(&gears));

        // Cables burn 1 plate a craft and yield 2, so the machine offers
        // 1.72 cable/s -- and now the output inserter is the bottleneck at 0.86.
        assert!(
            approx(score(&cable), INSERTER_RATE),
            "cable: {}",
            score(&cable)
        );
    }

    /// The machine's own rate is a ceiling. The reference caps its assembler at
    /// a ratio of 1.0 and never reads `crafting_time`, so flooding it with input
    /// buys output for free; here it buys nothing.
    #[test]
    fn flooding_an_assembler_does_not_beat_its_crafting_speed() {
        // Bolt the machine straight onto the source: unlimited plates, no
        // inserter throttling the input.
        let mut g = Grid::new(4, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, assembler(Item::IronGear));
        g.set(2, 0, inserter(Direction::East));
        g.set(3, 0, anchor(Entity::Sink, Item::IronGear));

        // An AM1 running a 0.5 s recipe crafts once a second however much you
        // shovel at it, so 1.0 gear/s leaves the machine and the output inserter
        // carries 0.86 of it. The machine and the inserter are the limits; the
        // supply never is. The reference's ratio model would take the free lunch.
        let recipe = Item::IronGear.recipe().expect("gears are craftable");
        assert!(approx(recipe.crafts_per_second(), 1.0));
        assert!(approx(score(&g), INSERTER_RATE), "{}", score(&g));
    }

    /// The reward hack the item filter exists to reject: belt the raw plate
    /// straight to a gear sink and never build the machine at all.
    #[test]
    fn bypassing_the_assembler_scores_zero() {
        let mut g = Grid::new(5, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, Cell::belt(Direction::East));
        g.set(3, 0, Cell::belt(Direction::East));
        g.set(4, 0, anchor(Entity::Sink, Item::IronGear));
        assert_eq!(score(&g), 0.0);
        assert!(!item_reaches_sink(&g));
    }

    /// A belt running past a machine's wall does not load it. Only an inserter
    /// does.
    #[test]
    fn a_belt_alongside_an_assembler_does_not_feed_it() {
        let mut g = Grid::new(4, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, assembler(Item::IronGear));
        g.set(3, 0, anchor(Entity::Sink, Item::IronGear));
        // The belt offers, the machine refuses, and nothing is crafted. (The
        // machine would also need an inserter to unload it.)
        assert_eq!(score(&g), 0.0);
    }

    /// An inserter picks up from the tile behind it, not from one beside it.
    #[test]
    fn an_inserter_does_not_grab_from_its_side() {
        let mut g = Grid::new(3, 2);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        // Faces north, so it reaches for (1,1) -- empty -- not the belt at (1,0).
        g.set(1, 1, inserter(Direction::North));
        g.set(2, 1, anchor(Entity::Sink, Item::IronPlate));
        assert_eq!(score(&g), 0.0);
    }

    /// Half a job is worth less than half the score. This is why the mean is a
    /// power mean and not an arithmetic one.
    #[test]
    fn an_ignored_sink_drags_the_score_down() {
        let mut g = Grid::new(3, 2);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, anchor(Entity::Sink, Item::IronPlate));
        g.set(2, 1, anchor(Entity::Sink, Item::IronPlate));

        let report = throughput(&g);
        assert_eq!(report.deliveries.len(), 2);
        // One sink at a full belt, one starved: ((sqrt(15) + 0)/2)^2 = 3.75.
        assert!(approx(report.score, 3.75), "{}", report.score);
        assert!(report.score < BELT_RATE);
        // ...and the binary metric calls this a win.
        assert!(item_reaches_sink(&g));
    }

    #[test]
    fn feeding_both_sinks_beats_feeding_one() {
        let mut one = Grid::new(3, 2);
        one.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        one.set(1, 0, Cell::belt(Direction::East));
        one.set(2, 0, anchor(Entity::Sink, Item::IronPlate));
        one.set(2, 1, anchor(Entity::Sink, Item::IronPlate));

        // Run a second belt out of the source along the bottom row.
        let mut both = one.clone();
        both.set(0, 1, Cell::belt(Direction::East));
        both.set(1, 1, Cell::belt(Direction::East));

        assert!(approx(score(&one), 3.75), "{}", score(&one));
        // A source is an unlimited supply: branching off it costs the first
        // branch nothing, so both sinks get a full belt.
        assert!(approx(score(&both), BELT_RATE), "{}", score(&both));
        assert!(score(&both) > score(&one));
    }

    /// A source touching a sink builds nothing and delivers an unbounded rate;
    /// it must not out-score a real factory.
    #[test]
    fn a_source_touching_a_sink_scores_zero() {
        let mut g = Grid::new(2, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, anchor(Entity::Sink, Item::IronPlate));
        assert_eq!(score(&g), 0.0);
    }

    #[test]
    fn an_underground_belt_carries_a_full_belt_across_a_gap() {
        let mut g = Grid::new(6, 1);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
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
        g.set(5, 0, anchor(Entity::Sink, Item::IronPlate));
        assert!(approx(score(&g), BELT_RATE));
    }

    /// A belt loop gets no topological turn, so it delivers nothing — but it
    /// only starves what it actually feeds. The reference zeroes the entire
    /// factory when a cycle exists anywhere; a live sink elsewhere still scores
    /// here.
    #[test]
    fn a_belt_loop_starves_only_what_it_feeds() {
        let mut g = Grid::new(4, 3);
        // A working run along the top.
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(1, 0, Cell::belt(Direction::East));
        g.set(2, 0, anchor(Entity::Sink, Item::IronPlate));
        // A disconnected 2x2 loop below, feeding a second sink.
        g.set(0, 1, Cell::belt(Direction::East));
        g.set(1, 1, Cell::belt(Direction::South));
        g.set(1, 2, Cell::belt(Direction::West));
        g.set(0, 2, Cell::belt(Direction::North));
        g.set(3, 1, anchor(Entity::Sink, Item::IronPlate));

        let report = throughput(&g);
        assert_eq!(report.deliveries.len(), 2);
        let fed = report.deliveries.iter().find(|d| d.at == (2, 0)).unwrap();
        let starved = report.deliveries.iter().find(|d| d.at == (3, 1)).unwrap();
        assert!(approx(fed.achieved, BELT_RATE));
        assert_eq!(starved.achieved, 0.0);
        assert!(
            report.score > 0.0,
            "the loop must not zero the whole factory"
        );
    }

    /// Every factory the curriculum generates is functional by construction, so
    /// the graded metric must agree with the binary one on all of them. If this
    /// ever fails, one of the two simulators is wrong about the same grid.
    #[test]
    fn every_generated_solution_scores_above_zero() {
        for &kind in LessonKind::all() {
            for seed in 0..200u64 {
                let Some(sample) = factory_gen::generate(kind, 11, seed) else {
                    continue;
                };
                assert!(
                    item_reaches_sink(&sample.solution),
                    "{kind:?} seed {seed}: generated solution is not functional"
                );
                let s = score(&sample.solution);
                assert!(
                    s > 0.0,
                    "{kind:?} seed {seed}: functional but scores {s}\n{}",
                    crate::textual::render(&sample.solution)
                );
                assert!(s.is_finite(), "{kind:?} seed {seed}: score is {s}");
            }
        }
    }
}
