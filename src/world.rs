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
}

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

    /// All cells consistent -> a well-formed factory.
    pub fn is_consistent(&self) -> bool {
        self.cells.iter().all(Cell::is_consistent)
    }
}
