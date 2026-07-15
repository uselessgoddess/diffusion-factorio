//! Compact ASCII rendering of a factory grid, for debugging and for making
//! inference output human-checkable (the reference has a richer YAML format in
//! `factorion_rs/src/textual.rs`; we keep a lightweight single-char view).

use crate::world::{Direction, Entity, Grid, Misc};

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
pub fn glyph(grid: &Grid, x: usize, y: usize) -> char {
    let c = grid.get(x, y);
    if c.is_empty() {
        match grid.anchor_at(x, y) {
            Some(anchor) if anchor != (x, y) => {
                return body(glyph(grid, anchor.0, anchor.1));
            }
            _ if grid.is_obstacle(x, y) => return '#',
            _ => {}
        }
    }
    let arrow = |d: Direction| match d {
        Direction::North => '^',
        Direction::East => '>',
        Direction::South => 'v',
        Direction::West => '<',
        Direction::None => '.',
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
}
