//! Compact ASCII rendering of a factory grid, for debugging and for making
//! inference output human-checkable (the reference has a richer YAML format in
//! `factorion_rs/src/textual.rs`; we keep a lightweight single-char view).

use crate::world::{Direction, Entity, Grid, Misc};

/// One glyph per cell. Belts/undergrounds render as direction arrows so the
/// flow is readable at a glance.
pub fn glyph(grid: &Grid, x: usize, y: usize) -> char {
    if grid.is_obstacle(x, y) && grid.get(x, y).is_empty() {
        return '#';
    }
    let c = grid.get(x, y);
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
}
