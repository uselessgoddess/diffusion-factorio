//! Grid / blueprint representation for the Factorio-inspired factory world.
//!
//! Design notes (see `docs/ANALYSIS.md` and `docs/DESIGN.md`):
//!
//! Borrowed from `beyarkay/factorion`:
//!   * A factory is a fixed-size 2D grid; each cell is described by several
//!     *categorical* channels (entity / direction / item / misc). Category ids
//!     are never fed to the network as raw ordinals — they are embedded or
//!     one-hot encoded, so no false ordering is imposed.
//!   * Multi-head consistency matters: a legal cell has consistent channels
//!     (e.g. an underground belt must carry a misc up/down tag). We therefore
//!     denoise all channels *jointly*.
//!
//! Rejected from `factorion`:
//!   * The `FOOTPRINT` channel is NOT part of the generative state. In the
//!     reference it once leaked the answer (only the correct placement cells
//!     were marked buildable). We keep obstacles in a *separate* conditioning
//!     channel that never encodes where entities should go.
//!   * The 89-item catalogue is trimmed to a small, extensible set — enough to
//!     exercise every output head without drowning rare classes.
//!   * **Stamping a multi-tile entity into every tile it covers.** The reference
//!     writes an assembler into all nine of its cells, which is natural for a PPO
//!     policy whose action is the single atomic "place a machine at (x, y)" — the
//!     env stamps, the agent never sees the nine. Our model emits every cell
//!     independently within a reveal round, so nine cells agreeing is luck, not
//!     learning. We store a machine **only at its top-left anchor** and leave the
//!     other tiles `Empty` but claimed. See [`Grid::anchor_at`] and
//!     [`Grid::footprints_are_legal`].

use serde::{Deserialize, Serialize};

/// Categorical channels that make up the *generative* state (the tokens the
/// diffusion model denoises). Order matters: it is the channel axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Entity = 0,
    Direction = 1,
    Item = 2,
    Misc = 3,
}

/// Number of categorical channels the model jointly denoises.
pub const N_CHANNELS: usize = 4;

/// Vocabulary size (number of real classes, *excluding* the diffusion MASK
/// token) for each channel, indexed by `Channel as usize`.
pub const VOCAB: [usize; N_CHANNELS] = [Entity::COUNT, Direction::COUNT, Item::COUNT, Misc::COUNT];

/// Entity kinds. `Empty` is id 0 so a blank grid is all-zeros.
///
/// `Source`/`Sink` are environment-provided anchors (inputs/outputs of the
/// factory); the generator always keeps them fixed, and the model conditions on
/// them but is not asked to invent them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Entity {
    Empty = 0,
    Source = 1,
    Sink = 2,
    TransportBelt = 3,
    UndergroundBelt = 4,
    Splitter = 5,
    Inserter = 6,
    Assembler = 7,
}

impl Entity {
    pub const COUNT: usize = 8;
    pub fn from_id(id: usize) -> Option<Self> {
        Some(match id {
            0 => Self::Empty,
            1 => Self::Source,
            2 => Self::Sink,
            3 => Self::TransportBelt,
            4 => Self::UndergroundBelt,
            5 => Self::Splitter,
            6 => Self::Inserter,
            7 => Self::Assembler,
            _ => return None,
        })
    }
    /// Whether this entity meaningfully carries a `Direction` (belts, inserters…).
    pub fn is_directional(self) -> bool {
        !matches!(self, Entity::Empty | Entity::Source | Entity::Sink)
    }

    /// Footprint in tiles, at the entity's default (north/south) facing.
    ///
    /// These are the vanilla prototype sizes, and they are not decoration.
    /// `blueprint.rs` has always anchored a real 3×3 `assembling-machine-1` and
    /// a real 2×1 `splitter` at the cell the world model called 1×1, so every
    /// blueprint we exported for a lesson containing either one placed entities
    /// on top of each other and Factorio refused to import it. The size lives
    /// here, once, so the simulator and the exporter cannot disagree about it.
    ///
    /// `Source`/`Sink` are our own anchors rather than vanilla prototypes, and
    /// stay 1×1: they mark where the factory's input and output *are*, and give
    /// the model nothing to lay out.
    pub fn size(self) -> (usize, usize) {
        match self {
            Entity::Assembler => (3, 3),
            Entity::Splitter => (2, 1),
            _ => (1, 1),
        }
    }

    /// Footprint `(width, height)` once rotated to face `direction`.
    ///
    /// A square footprint ignores facing — a 3×3 assembler covers the same nine
    /// tiles whichever way it points, and only its recipe I/O cares about the
    /// direction. A 2×1 splitter laid out east-west becomes 1×2 when it faces
    /// east or west. The reference guards the same case in `entity_tiles`, which
    /// skips the swap when `width == height`.
    pub fn footprint(self, direction: Direction) -> (usize, usize) {
        let (w, h) = self.size();
        match direction {
            Direction::East | Direction::West if w != h => (h, w),
            _ => (w, h),
        }
    }
}

/// Longest side of any [`Entity::size`]. Bounds how far up and left of a tile
/// an anchor covering it can sit, which is what makes [`Grid::anchor_at`] a
/// constant-time scan instead of a sweep of the whole grid.
pub const MAX_FOOTPRINT: usize = 3;

/// Facing. `None` is used by non-directional cells (empty / source / sink) so
/// the direction channel is unambiguous for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Direction {
    None = 0,
    North = 1,
    East = 2,
    South = 3,
    West = 4,
}

impl Direction {
    pub const COUNT: usize = 5;
    pub fn from_id(id: usize) -> Option<Self> {
        Some(match id {
            0 => Self::None,
            1 => Self::North,
            2 => Self::East,
            3 => Self::South,
            4 => Self::West,
            _ => return None,
        })
    }
    /// Unit step (dx, dy) this direction points to. `None` -> (0, 0).
    pub fn delta(self) -> (i32, i32) {
        match self {
            Direction::None => (0, 0),
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
        }
    }
    pub fn opposite(self) -> Self {
        match self {
            Direction::None => Direction::None,
            Direction::North => Direction::South,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }
}

/// How an assembler turns one item into another.
///
/// `crafting_time` is seconds per craft *before* the machine's crafting speed
/// multiplier, matching the wiki's Base-game value. The real rate of a machine
/// running this recipe is `crafting_speed / crafting_time` crafts per second —
/// see [`ASSEMBLER_CRAFTING_SPEED`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recipe {
    /// What one craft consumes. A machine runs at the rate of its **scarcest**
    /// input, so an assembler drowning in iron and starved of cable makes
    /// nothing — which is the whole reason a recipe is a list and not a field.
    pub ingredients: &'static [Ingredient],
    /// Units of the product yielded per craft.
    pub output_qty: f64,
    /// Seconds per craft at crafting speed 1.
    pub crafting_time: f64,
}

/// One input to a recipe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ingredient {
    pub item: Item,
    /// Units consumed per craft.
    pub qty: f64,
}

impl Recipe {
    /// Crafts per second this recipe achieves in an assembler that is never
    /// starved of input.
    pub fn crafts_per_second(&self) -> f64 {
        ASSEMBLER_CRAFTING_SPEED / self.crafting_time
    }

    /// Items per second the assembler emits when never starved.
    pub fn max_output_rate(&self) -> f64 {
        self.crafts_per_second() * self.output_qty
    }

    /// Items per second of `item` needed to keep the machine saturated, or `0`
    /// if this recipe does not use it.
    pub fn max_input_rate(&self, item: Item) -> f64 {
        self.crafts_per_second() * self.qty_of(item)
    }

    /// Units of `item` one craft consumes; `0` when the recipe does not use it.
    pub fn qty_of(&self, item: Item) -> f64 {
        self.ingredients
            .iter()
            .find(|i| i.item == item)
            .map_or(0.0, |i| i.qty)
    }

    /// Does this recipe need more than one distinct input?
    pub fn is_multi_input(&self) -> bool {
        self.ingredients.len() > 1
    }
}

/// Crafting speed of the assembler we model (vanilla Assembling Machine 1).
///
/// The reference project's throughput engine never reads this, nor
/// `crafting_time`: `factorion_rs/src/entities.rs` models an assembler as a
/// pass-through *ratio* capped at 1.0, so in their engine a machine can emit as
/// fast as it is fed. That makes an assembler invisible to the score — it can
/// never be the bottleneck, which is exactly what an assembler usually *is* in
/// a real factory. We model the machine's real rate instead.
pub const ASSEMBLER_CRAFTING_SPEED: f64 = 0.5;

/// Recipe / filtered-item catalogue. `None` when a cell has no associated item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Item {
    None = 0,
    IronPlate = 1,
    CopperPlate = 2,
    IronGear = 3,
    CopperCable = 4,
    GreenCircuit = 5,
}

impl Item {
    pub const COUNT: usize = 6;
    pub fn from_id(id: usize) -> Option<Self> {
        Some(match id {
            0 => Self::None,
            1 => Self::IronPlate,
            2 => Self::CopperPlate,
            3 => Self::IronGear,
            4 => Self::CopperCable,
            5 => Self::GreenCircuit,
            _ => return None,
        })
    }

    /// Everything an assembler must be fed to craft this item. Empty for a raw
    /// item that no recipe produces.
    ///
    /// This is the one place the recipe graph is defined — both the lesson
    /// generator and the simulator read it, so a factory can never be generated
    /// that the simulator would call broken.
    pub fn ingredients(self) -> &'static [Ingredient] {
        self.recipe().map_or(&[], |r| r.ingredients)
    }

    /// The recipe that produces this item, or `None` for a raw item.
    ///
    /// Quantities and `crafting_time` are the vanilla (Base game) values, which
    /// is what makes a *graded* throughput score meaningful: gears and cables
    /// both craft in 0.5 s, but one consumes 2 plates to yield 1 item and the
    /// other consumes 1 to yield 2, so an identical layout delivers a different
    /// items/s depending on the recipe. A model that only knows "connected"
    /// cannot see that difference.
    ///
    /// These are the vanilla recipes as the wiki states them, including the
    /// electronic circuit's **two** inputs. Until the 3×3 footprint landed they
    /// could not be: a 1×1 machine has one tile in front and one behind, so a
    /// second input had nowhere to go, and every recipe here was forced down to a
    /// single ingredient. A 3×3 machine has twelve perimeter slots
    /// ([`Grid::perimeter`]) and can be fed from as many sides as a recipe needs,
    /// so the simplification is gone.
    ///
    /// A two-input recipe is a different *kind* of problem, not a bigger one. A
    /// single-input line is a path: connect A to B. A circuit needs two paths
    /// that arrive at the same machine and neither may be sacrificed for the
    /// other, and — because a craft consumes 3 cable per 1 plate — feeding them
    /// equally is wrong. The layout that scores well has to be unbalanced on
    /// purpose, which nothing in a connectivity metric can express.
    pub fn recipe(self) -> Option<Recipe> {
        Some(match self {
            // 2 iron plate -> 1 iron gear wheel, 0.5 s.
            Self::IronGear => Recipe {
                ingredients: &[Ingredient {
                    item: Self::IronPlate,
                    qty: 2.0,
                }],
                output_qty: 1.0,
                crafting_time: 0.5,
            },
            // 1 copper plate -> 2 copper cable, 0.5 s.
            Self::CopperCable => Recipe {
                ingredients: &[Ingredient {
                    item: Self::CopperPlate,
                    qty: 1.0,
                }],
                output_qty: 2.0,
                crafting_time: 0.5,
            },
            // 1 iron plate + 3 copper cable -> 1 electronic circuit, 0.5 s.
            // https://wiki.factorio.com/Electronic_circuit
            Self::GreenCircuit => Recipe {
                ingredients: &[
                    Ingredient {
                        item: Self::IronPlate,
                        qty: 1.0,
                    },
                    Ingredient {
                        item: Self::CopperCable,
                        qty: 3.0,
                    },
                ],
                output_qty: 1.0,
                crafting_time: 0.5,
            },
            Self::None | Self::IronPlate | Self::CopperPlate => return None,
        })
    }

    /// Items an assembler can be given as a recipe.
    pub fn craftable() -> [Self; 3] {
        [Self::IronGear, Self::CopperCable, Self::GreenCircuit]
    }

    /// The craftables a single feed line can satisfy. The lessons that lay one
    /// source into one machine can only teach these; a two-input recipe needs a
    /// lesson that builds two feeds ([`crate::factory_gen::LessonKind`]).
    pub fn single_input_craftable() -> Vec<Self> {
        Self::craftable()
            .into_iter()
            .filter(|i| i.recipe().is_some_and(|r| !r.is_multi_input()))
            .collect()
    }
}

/// Misc per-cell state. Currently the underground-belt endpoint tag, mirroring
/// factorion's MISC channel. `None` for everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Misc {
    None = 0,
    UndergroundDown = 1, // entrance (goes under)
    UndergroundUp = 2,   // exit (comes up)
}

impl Misc {
    pub const COUNT: usize = 3;
    pub fn from_id(id: usize) -> Option<Self> {
        Some(match id {
            0 => Self::None,
            1 => Self::UndergroundDown,
            2 => Self::UndergroundUp,
            _ => return None,
        })
    }
}

/// One cell = the categorical value of every channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub entity: Entity,
    pub direction: Direction,
    pub item: Item,
    pub misc: Misc,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            entity: Entity::Empty,
            direction: Direction::None,
            item: Item::None,
            misc: Misc::None,
        }
    }
}

impl Cell {
    pub fn belt(direction: Direction) -> Self {
        Self {
            entity: Entity::TransportBelt,
            direction,
            ..Default::default()
        }
    }
    pub fn is_empty(&self) -> bool {
        self.entity == Entity::Empty
    }
    /// Category id for a given channel.
    pub fn channel_id(&self, ch: Channel) -> usize {
        match ch {
            Channel::Entity => self.entity as usize,
            Channel::Direction => self.direction as usize,
            Channel::Item => self.item as usize,
            Channel::Misc => self.misc as usize,
        }
    }
    /// Whether channels are mutually consistent (a legal cell). Used by
    /// validation metrics to score whether a decoded factory is well-formed.
    pub fn is_consistent(&self) -> bool {
        // Directional entities must face somewhere; non-directional must not.
        if self.entity.is_directional() != (self.direction != Direction::None) {
            return false;
        }
        // Only underground belts carry a misc up/down tag.
        let has_misc = self.misc != Misc::None;
        if has_misc != (self.entity == Entity::UndergroundBelt) {
            return false;
        }
        // Only item-bearing entities carry an item tag: assemblers (recipe) and
        // the source/sink anchors (which item they provide / accept).
        let has_item = self.item != Item::None;
        let item_bearing = matches!(
            self.entity,
            Entity::Assembler | Entity::Source | Entity::Sink
        );
        if has_item && !item_bearing {
            return false;
        }
        true
    }
}

/// A fixed-size factory grid stored row-major (`idx = y * width + x`).
///
/// `obstacle` is the separate conditioning channel (buildable footprint):
/// `true` = permanently blocked, entities may not be placed there. It is an
/// *input* to the model, never a generative channel, and by default a fresh
/// grid has no obstacles (avoiding the footprint data leak).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
    pub obstacle: Vec<bool>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            cells: vec![Cell::default(); n],
            obstacle: vec![false; n],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Cell {
        self.cells[self.idx(x, y)]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, cell: Cell) {
        let i = self.idx(x, y);
        self.cells[i] = cell;
    }
    #[inline]
    pub fn is_obstacle(&self, x: usize, y: usize) -> bool {
        self.obstacle[self.idx(x, y)]
    }
    pub fn set_obstacle(&mut self, x: usize, y: usize, v: bool) {
        let i = self.idx(x, y);
        self.obstacle[i] = v;
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The tiles covered by the entity anchored at `(x, y)`, which is its
    /// top-left cell. Tiles that fall off the grid are still yielded, so a
    /// caller can tell "hangs over the edge" from "fits" — see
    /// [`Self::footprints_are_legal`].
    ///
    /// An empty cell reports itself, since [`Entity::Empty`] is 1×1. Ask
    /// [`Cell::is_empty`] first if that distinction matters.
    pub fn footprint_at(&self, x: usize, y: usize) -> Vec<(i32, i32)> {
        let cell = self.get(x, y);
        let (w, h) = cell.entity.footprint(cell.direction);
        let (x, y) = (x as i32, y as i32);
        (0..h as i32)
            .flat_map(|dy| (0..w as i32).map(move |dx| (x + dx, y + dy)))
            .collect()
    }

    /// The anchor of the entity covering `(x, y)`, or `None` if none does.
    ///
    /// For a 1×1 entity this is `(x, y)` itself; for one of the eight tiles
    /// around a 3×3 assembler's anchor it is that anchor — the only cell that
    /// stores the entity at all. **Every read of the form "what is standing on
    /// this tile?" has to come through here.** [`Self::get`] answers "what is
    /// *stored* in this cell", and for eight ninths of an assembler the honest
    /// answer to that is `Empty`, which is exactly the hole a consumer must not
    /// fall into.
    ///
    /// On a grid whose footprints overlap (which [`Self::is_consistent`]
    /// rejects) this reports the first anchor in scan order.
    pub fn anchor_at(&self, x: usize, y: usize) -> Option<(usize, usize)> {
        let first = |v: usize| v.saturating_sub(MAX_FOOTPRINT - 1);
        (first(y)..=y)
            .flat_map(|ay| (first(x)..=x).map(move |ax| (ax, ay)))
            .find(|&(ax, ay)| {
                !self.get(ax, ay).is_empty()
                    && self.footprint_at(ax, ay).contains(&(x as i32, y as i32))
            })
    }

    /// Tiles orthogonally adjacent to the footprint of the entity anchored at
    /// `(x, y)`: every slot an inserter could stand in to load or unload it.
    ///
    /// A 1×1 entity has four. A 3×3 assembler has twelve — its 16-tile ring
    /// minus the four diagonal corners, which touch it at a point and cannot
    /// reach it. Those twelve slots are the reason multi-tile machines are worth
    /// the trouble: a 1×1 machine has room for one input and one output, so a
    /// recipe needing two different items delivered can only be expressed once
    /// the machine has a perimeter. The reference enumerates the same ring as
    /// `PERIM_SLOTS`.
    pub fn perimeter(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let cell = self.get(x, y);
        let (w, h) = cell.entity.footprint(cell.direction);
        let (x, y, w, h) = (x as i32, y as i32, w as i32, h as i32);
        let sides = (0..h).flat_map(|dy| [(x - 1, y + dy), (x + w, y + dy)]);
        let ends = (0..w).flat_map(|dx| [(x + dx, y - 1), (x + dx, y + h)]);
        sides
            .chain(ends)
            .filter(|&(nx, ny)| self.in_bounds(nx, ny))
            .map(|(nx, ny)| (nx as usize, ny as usize))
            .collect()
    }

    /// Do the footprints tile the grid legally? Every entity must fit on the
    /// grid, must not cover an obstacle, and must not overlap another entity.
    ///
    /// This is the first rule in this file that a single cell cannot answer, and
    /// it exists because [`Entity::size`] made an assembler 3×3. The eight cells
    /// around its anchor are *stored* `Empty`, but they are not free, and a
    /// factory that belts a lane straight through the middle of a machine is not
    /// a factory — it is a grid that renders plausibly and imports as nothing.
    ///
    /// Storing the entity only at its anchor, rather than stamping it across all
    /// nine cells the way the reference does, is what keeps this cheap for a
    /// diffusion model: the model commits cells independently within a reveal
    /// round, so "these nine cells must all agree" is a constraint it can only
    /// satisfy by luck, while "leave the machine's shadow alone" is a constraint
    /// whose correct answer is the `Empty` it already predicts everywhere else.
    pub fn footprints_are_legal(&self) -> bool {
        let mut claimed = vec![false; self.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y).is_empty() {
                    continue;
                }
                for (tx, ty) in self.footprint_at(x, y) {
                    if !self.in_bounds(tx, ty) {
                        return false;
                    }
                    let i = self.idx(tx as usize, ty as usize);
                    if self.obstacle[i] || claimed[i] {
                        return false;
                    }
                    claimed[i] = true;
                }
            }
        }
        true
    }

    /// Every cell consistent *and* every footprint legal -> a well-formed
    /// factory.
    pub fn is_consistent(&self) -> bool {
        self.cells.iter().all(Cell::is_consistent) && self.footprints_are_legal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assembler() -> Cell {
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::IronGear,
            misc: Misc::None,
        }
    }

    fn splitter(direction: Direction) -> Cell {
        Cell {
            entity: Entity::Splitter,
            direction,
            ..Default::default()
        }
    }

    #[test]
    fn a_square_footprint_ignores_facing_and_a_splitter_rotates() {
        for d in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            assert_eq!(Entity::Assembler.footprint(d), (3, 3));
            assert_eq!(Entity::TransportBelt.footprint(d), (1, 1));
        }
        assert_eq!(Entity::Splitter.footprint(Direction::North), (2, 1));
        assert_eq!(Entity::Splitter.footprint(Direction::South), (2, 1));
        assert_eq!(Entity::Splitter.footprint(Direction::East), (1, 2));
        assert_eq!(Entity::Splitter.footprint(Direction::West), (1, 2));
    }

    /// The whole point of the anchor representation: eight of an assembler's
    /// nine cells are stored `Empty`, and every consumer must still see a
    /// machine standing on them.
    #[test]
    fn body_tiles_resolve_to_the_anchor_though_they_read_as_empty() {
        let mut g = Grid::new(6, 6);
        g.set(1, 1, assembler());
        for y in 1..4 {
            for x in 1..4 {
                assert!(
                    (x, y) == (1, 1) || g.get(x, y).is_empty(),
                    "({x},{y}) should be stored empty"
                );
                assert_eq!(g.anchor_at(x, y), Some((1, 1)), "({x},{y}) is the machine");
            }
        }
        assert_eq!(
            g.anchor_at(4, 4),
            None,
            "diagonal corner is outside the 3×3"
        );
        assert_eq!(g.anchor_at(0, 1), None, "left of the anchor is free");
    }

    #[test]
    fn a_rotated_splitter_covers_the_tile_below_not_beside() {
        let mut g = Grid::new(4, 4);
        g.set(1, 1, splitter(Direction::East));
        assert_eq!(g.anchor_at(1, 2), Some((1, 1)));
        assert_eq!(g.anchor_at(2, 1), None);
    }

    /// Twelve slots, not sixteen: the diagonal corners touch the machine at a
    /// point, and an inserter cannot reach through a point.
    #[test]
    fn a_machine_perimeter_is_the_ring_minus_its_corners() {
        let mut g = Grid::new(9, 9);
        g.set(3, 3, assembler());
        let ring = g.perimeter(3, 3);
        assert_eq!(ring.len(), 12);
        for corner in [(2, 2), (6, 2), (2, 6), (6, 6)] {
            assert!(
                !ring.contains(&corner),
                "{corner:?} only touches diagonally"
            );
        }
        for slot in [(2, 3), (2, 4), (2, 5), (6, 3), (3, 2), (4, 6)] {
            assert!(ring.contains(&slot), "{slot:?} should be a loading slot");
        }

        g.set(0, 0, Cell::belt(Direction::East));
        assert_eq!(
            g.perimeter(0, 0).len(),
            2,
            "a corner cell has two neighbours"
        );
    }

    /// The bug this representation exists to fix. Before footprints, this grid
    /// was "consistent" — every cell was individually well-formed — and it
    /// exported to a blueprint whose belt sat inside the machine.
    #[test]
    fn an_entity_in_a_machines_shadow_is_not_a_well_formed_factory() {
        let mut g = Grid::new(6, 6);
        g.set(1, 1, assembler());
        assert!(g.is_consistent(), "a lone assembler with room is fine");

        g.set(2, 2, Cell::belt(Direction::East));
        assert!(
            g.cells.iter().all(Cell::is_consistent),
            "every cell is still individually legal — only the grid knows better"
        );
        assert!(!g.is_consistent(), "the belt is inside the assembler");
    }

    #[test]
    fn a_footprint_may_not_hang_off_the_grid() {
        let mut g = Grid::new(6, 6);
        g.set(3, 3, assembler());
        assert!(g.is_consistent(), "3..6 is the last column that fits");
        g.set(3, 3, Cell::default());
        g.set(4, 3, assembler());
        assert!(!g.is_consistent(), "the machine's right column is off-grid");
    }

    /// Obstacles are checked against the whole footprint, not just the anchor:
    /// a machine placed one tile clear of a rock still has eight other cells.
    #[test]
    fn a_footprint_may_not_cover_an_obstacle() {
        let mut g = Grid::new(6, 6);
        g.set(1, 1, assembler());
        g.set_obstacle(3, 3, true);
        assert!(!g.is_consistent());
    }

    #[test]
    fn two_machines_may_not_overlap() {
        let mut g = Grid::new(8, 8);
        g.set(1, 1, assembler());
        g.set(4, 1, assembler());
        assert!(g.is_consistent(), "abutting machines share no tile");
        g.set(4, 1, Cell::default());
        g.set(3, 1, assembler());
        assert!(!g.is_consistent(), "they share the column at x=3");
    }
}
