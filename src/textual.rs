//! Compact ASCII rendering of a factory grid, for debugging and for making
//! inference output human-checkable (the reference has a richer YAML format in
//! `factorion_rs/src/textual.rs`; we keep a lightweight single-char view).

use crate::world::{Cell, Direction, Entity, Grid, Misc};

/// One glyph per cell. Belts/undergrounds render as direction arrows so the
/// flow is readable at a glance.
///
/// A machine is stored only at its top-left anchor, so its other tiles read as
/// `Empty` even though they are covered. Drawing those as bare floor would print
/// a 3×3 assembler as a lone `A` — the exact 1×1 shape the world model used to
/// have and deliberately no longer does, and the reader would have no way to see
/// the difference. They are drawn as the **lowercase** of their anchor's glyph
/// instead: `A` is the machine, `a` is its body.
///
/// A covered tile that holds an entity of its own is a violation
/// ([`Grid::footprints_are_legal`] rejects it), and it keeps its own glyph so
/// that it stays visible. Predictions are drawn through here too, and a view
/// that quietly redrew a model's overlap as tidy machine body would hide the
/// mistake it exists to show.
///
/// The same rule decides the other two glyphs, which say that a tile is wrong
/// rather than what is on it:
///
/// * `!` — something is built on an obstacle. The `#` cannot simply give way to
///   the entity: terrain is the one thing on the grid the model does not get to
///   choose, so a tile that lost its `#` would read as a legal build.
/// * `?` — the entity is not a thing that can exist: a belt facing nowhere, an
///   underground that is neither end of a tunnel. [`Cell::is_consistent`] is what
///   rejects these, and drawing a directionless belt as floor is how a grid full
///   of them can still look like a tidy answer.
pub fn glyph(grid: &Grid, x: usize, y: usize) -> char {
    match (built(grid, x, y), grid.is_obstacle(x, y)) {
        (Some(_), true) => '!',
        (Some(glyph), false) => glyph,
        (None, true) => '#',
        (None, false) => '.',
    }
}

/// The glyph for whatever occupies a tile, or `None` if the tile is bare floor.
///
/// Obstacles are not consulted: they are terrain under the tile rather than
/// something built on it, and keeping the two apart is what lets [`glyph`] show
/// a tile where both are true.
fn built(grid: &Grid, x: usize, y: usize) -> Option<char> {
    let cell = grid.get(x, y);
    if cell.is_empty() {
        return match grid.anchor_at(x, y) {
            Some(anchor) if anchor != (x, y) => Some(body(entity(grid.get(anchor.0, anchor.1)))),
            _ => None,
        };
    }
    Some(entity(cell))
}

/// The glyph for a cell that holds something, judged on the cell alone.
fn entity(c: Cell) -> char {
    let arrow = |d: Direction| match d {
        Direction::North => '^',
        Direction::East => '>',
        Direction::South => 'v',
        Direction::West => '<',
        // Not floor: a belt that carries in no direction is not a belt, and the
        // exporter refuses it. See [`glyph`].
        Direction::None => '?',
    };
    match c.entity {
        Entity::Empty => '.',
        Entity::Source => 'S',
        Entity::Sink => 'K',
        Entity::TransportBelt => arrow(c.direction),
        Entity::UndergroundBelt => match c.misc {
            Misc::UndergroundDown => 'd',
            Misc::UndergroundUp => 'u',
            Misc::None => '?',
        },
        Entity::Splitter => 'Y',
        Entity::Inserter => 'i',
        Entity::Assembler => 'A',
    }
}

/// The glyph for a tile a machine covers, given the glyph of the machine
/// itself. Lowercasing keeps the two legible as one object while still telling
/// you where the anchor is — which matters, because the anchor is the only tile
/// that holds anything.
fn body(anchor: char) -> char {
    anchor.to_ascii_lowercase()
}

/// Render the whole grid as a multi-line string.
pub fn render(grid: &Grid) -> String {
    let mut s = String::with_capacity((grid.width + 1) * grid.height);
    for y in 0..grid.height {
        for x in 0..grid.width {
            s.push(glyph(grid, x, y));
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Cell, Direction, Entity};

    #[test]
    fn renders_belt_line() {
        let mut g = Grid::new(4, 1);
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
        g.set(
            3,
            0,
            Cell {
                entity: Entity::Sink,
                ..Default::default()
            },
        );
        assert_eq!(render(&g), "S>>K\n");
    }

    /// The whole point of the view: an assembler must not read as one tile,
    /// because it is not one tile.
    #[test]
    fn a_machine_draws_the_tiles_it_covers() {
        let mut g = Grid::new(5, 3);
        g.set(
            0,
            1,
            Cell {
                entity: Entity::Source,
                ..Default::default()
            },
        );
        g.set(
            1,
            0,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(
            4,
            1,
            Cell {
                entity: Entity::Sink,
                ..Default::default()
            },
        );
        assert_eq!(render(&g), ".Aaa.\nSaaaK\n.aaa.\n");
    }

    /// An entity standing inside a machine cannot be built, so the view has to
    /// keep showing it rather than tidy it away into body glyphs.
    #[test]
    fn an_entity_overlapping_a_machine_stays_visible() {
        let mut g = Grid::new(3, 3);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set(1, 1, Cell::belt(Direction::East));
        assert!(!g.is_consistent(), "a belt inside a machine is not legal");
        assert_eq!(render(&g), "Aaa\na>a\naaa\n");
    }

    /// The cell the issue's log kept refusing to export, drawn. It used to render
    /// as `.`, so a grid the exporter rejected outright printed as clean floor and
    /// the reader had no way to see what the exporter was objecting to.
    #[test]
    fn a_belt_facing_nowhere_does_not_draw_as_floor() {
        let mut g = Grid::new(3, 1);
        g.set(
            1,
            0,
            Cell {
                entity: Entity::TransportBelt,
                direction: Direction::None,
                ..Default::default()
            },
        );
        assert!(!g.is_consistent(), "a belt facing nowhere is not legal");
        assert_eq!(render(&g), ".?.\n");
        // And the tile it sits on is not floor, whatever it looked like.
        assert_ne!(glyph(&g, 1, 0), glyph(&g, 0, 0));
    }

    /// An underground belt that is neither end of a tunnel is the same class of
    /// non-thing, and has always drawn as `?`. Stated here so the two stay in
    /// step: both mean "this is not an entity", not "this is floor".
    #[test]
    fn an_underground_that_is_neither_end_draws_as_broken_too() {
        let mut g = Grid::new(1, 1);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::UndergroundBelt,
                direction: Direction::East,
                misc: Misc::None,
                ..Default::default()
            },
        );
        assert!(!g.is_consistent());
        assert_eq!(render(&g), "?\n");
    }

    /// Terrain is the one thing on the grid the model does not choose, so an
    /// entity standing on it is a violation ([`Grid::footprints_are_legal`]
    /// rejects it). The `#` used to give way to the entity's own glyph, which
    /// drew the illegal build as an ordinary one on bare floor.
    #[test]
    fn an_entity_built_on_an_obstacle_does_not_hide_it() {
        let mut g = Grid::new(3, 1);
        g.set_obstacle(1, 0, true);
        assert_eq!(render(&g), ".#.\n");

        g.set(1, 0, Cell::belt(Direction::East));
        assert!(!g.is_consistent(), "a belt on an obstacle is not legal");
        assert_eq!(render(&g), ".!.\n");
    }

    /// Same rule, reached through a machine's body rather than its anchor: the
    /// footprint is implied, so this tile reads as `Empty` and used to be drawn
    /// as tidy machine body over the obstacle it illegally covers.
    #[test]
    fn an_obstacle_under_a_machines_body_stays_visible() {
        let mut g = Grid::new(3, 3);
        g.set(
            0,
            0,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                ..Default::default()
            },
        );
        g.set_obstacle(2, 2, true);
        assert!(!g.is_consistent(), "a machine may not cover an obstacle");
        assert_eq!(render(&g), "Aaa\naaa\naa!\n");
    }
}
