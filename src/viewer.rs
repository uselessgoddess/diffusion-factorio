//! Draw a factory the way it looks, not the way it is stored.
//!
//! Issue #9: *"я так как не особо вижу что там обучается и что модель хорошо
//! инферит"* — "I can't really see what's being trained or whether the model
//! infers well". The existing report answers that with [`crate::textual`]
//! glyphs: an assembler is `A`, its eight body tiles are `a`, a belt is `>`.
//! That is a good debugger and a poor viewer. A wall of letters cannot show you
//! that a machine is a 3×3 block, that two inserters face into it, or that a
//! belt lane runs past without ever touching it — which are exactly the things a
//! human is looking for when they ask "is this factory any good?".
//!
//! So this module renders a [`Grid`] to SVG: one rect per entity at its **real
//! footprint**, drawn from [`Entity::footprint`], which is the same table
//! `blueprint.rs` exports from and `world.rs` validates against. A machine that
//! is 3×3 in Factorio is 3×3 here, and a viewer that disagreed with the exporter
//! would be reintroducing the very bug this rendering exists to make visible.
//!
//! SVG rather than a canvas script or a PNG: it is text, so it embeds directly
//! in the offline reports (`docs/OBSERVABILITY.md` — no network, no account, no
//! build step), it stays sharp at any zoom, and every tile carries a `<title>`
//! so hovering tells you what the model actually committed there.
//!
//! This is deliberately *not* a Factorio-accurate render. It is a schematic:
//! flat colours, an arrow for facing, a short item tag. The authoritative view
//! of a factory is the blueprint string from [`crate::blueprint`] pasted into
//! the real game — see `docs/VIEWER.md` for why that is the primary path and
//! this is the triage tool that sits in front of it.

use crate::world::{Direction, Entity, Grid, Item, Misc};
use std::fmt::Write as _;

/// Side of one tile, in SVG units.
const TILE: usize = 22;

/// Fill colour per entity, roughly tracking the game's own palette so the
/// picture is readable at a glance by someone who knows Factorio: belts yellow,
/// inserters blue, machines teal, our own I/O anchors green and red.
fn fill(entity: Entity) -> &'static str {
    match entity {
        Entity::Empty => "#1b1f24",
        Entity::Source => "#2f7d4f",
        Entity::Sink => "#a4373a",
        Entity::TransportBelt => "#b8912f",
        Entity::UndergroundBelt => "#7d5f1f",
        Entity::Splitter => "#c2a04a",
        Entity::Inserter => "#3f6ea8",
        Entity::Assembler => "#2f7f86",
    }
}

/// Short tag drawn on an entity that carries an item, so a source, a sink and a
/// machine's recipe are told apart without a legend.
fn item_tag(item: Item) -> &'static str {
    match item {
        Item::None => "",
        Item::IronPlate => "Fe",
        Item::CopperPlate => "Cu",
        Item::IronGear => "gear",
        Item::CopperCable => "wire",
        Item::GreenCircuit => "chip",
    }
}

/// The arrow glyph for a facing, or `None` for the non-directional cells.
fn arrow(direction: Direction) -> Option<&'static str> {
    match direction {
        Direction::North => Some("▲"),
        Direction::East => Some("▶"),
        Direction::South => Some("▼"),
        Direction::West => Some("◀"),
        Direction::None => None,
    }
}

/// What a tile is, in words, for the hover text. This is the one place the
/// viewer says something the picture cannot.
fn describe(grid: &Grid, x: usize, y: usize) -> String {
    if grid.is_obstacle(x, y) {
        return format!("({x}, {y}) obstacle");
    }
    let cell = grid.get(x, y);
    if cell.is_empty() {
        return match grid.anchor_at(x, y) {
            Some((ax, ay)) => format!(
                "({x}, {y}) covered by the {:?} at ({ax}, {ay})",
                grid.get(ax, ay).entity
            ),
            None => format!("({x}, {y}) empty"),
        };
    }
    let mut s = format!("({x}, {y}) {:?}", cell.entity);
    if let Some(a) = arrow(cell.direction) {
        let _ = write!(s, " facing {:?} {a}", cell.direction);
    }
    if cell.item != Item::None {
        let _ = write!(s, ", {:?}", cell.item);
    }
    match cell.misc {
        Misc::UndergroundDown => s.push_str(", entrance"),
        Misc::UndergroundUp => s.push_str(", exit"),
        Misc::None => {}
    }
    s
}

/// Render `grid` as a standalone `<svg>` element.
///
/// Entities are drawn once, at their anchor, spanning every tile they cover.
/// Body tiles are therefore *not* drawn separately — they are already inside
/// the machine's rect, which is the whole point: the picture shows one 3×3
/// object where the storage holds one anchor and eight claimed blanks.
pub fn grid_to_svg(grid: &Grid) -> String {
    let (w, h) = (grid.width * TILE, grid.height * TILE);
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" \
         class=\"factory\" role=\"img\">"
    );

    // Ground first: one flat rect, then the tile lattice over it, so the grid
    // reads as a floor rather than as a pile of squares.
    let _ = write!(svg, "<rect width=\"{w}\" height=\"{h}\" fill=\"#1b1f24\"/>");
    for i in 0..=grid.width {
        let x = i * TILE;
        let _ = write!(
            svg,
            "<line x1=\"{x}\" y1=\"0\" x2=\"{x}\" y2=\"{h}\" stroke=\"#2b3138\" stroke-width=\"1\"/>"
        );
    }
    for i in 0..=grid.height {
        let y = i * TILE;
        let _ = write!(
            svg,
            "<line x1=\"0\" y1=\"{y}\" x2=\"{w}\" y2=\"{y}\" stroke=\"#2b3138\" stroke-width=\"1\"/>"
        );
    }

    for y in 0..grid.height {
        for x in 0..grid.width {
            if grid.is_obstacle(x, y) {
                let (px, py) = (x * TILE, y * TILE);
                let _ = write!(
                    svg,
                    "<rect x=\"{px}\" y=\"{py}\" width=\"{TILE}\" height=\"{TILE}\" \
                     fill=\"#3a3f45\"/><title>{}</title>",
                    describe(grid, x, y)
                );
            }
        }
    }

    for y in 0..grid.height {
        for x in 0..grid.width {
            let cell = grid.get(x, y);
            if cell.is_empty() {
                continue;
            }
            svg.push_str(&entity_svg(grid, x, y));
        }
    }

    svg.push_str("</svg>");
    svg
}

/// One entity, drawn at its anchor across its whole footprint.
fn entity_svg(grid: &Grid, x: usize, y: usize) -> String {
    let cell = grid.get(x, y);
    let (fw, fh) = cell.entity.footprint(cell.direction);
    let (px, py) = (x * TILE, y * TILE);
    let (w, h) = (fw * TILE, fh * TILE);
    let (cx, cy) = (px + w / 2, py + h / 2);

    let mut s = format!("<g><title>{}</title>", describe(grid, x, y));
    let _ = write!(
        s,
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"3\" fill=\"{}\" \
         stroke=\"#0d1013\" stroke-width=\"1.5\"/>",
        px + 1,
        py + 1,
        w.saturating_sub(2),
        h.saturating_sub(2),
        fill(cell.entity)
    );

    // An underground belt's whole meaning is which end it is, so it gets a
    // letter rather than an arrow; everything else directional gets the arrow.
    let mark = match cell.entity {
        Entity::UndergroundBelt => match cell.misc {
            Misc::UndergroundDown => Some("⤓"),
            Misc::UndergroundUp => Some("⤒"),
            Misc::None => Some("?"),
        },
        _ => arrow(cell.direction),
    };
    if let Some(mark) = mark {
        let _ = write!(
            s,
            "<text x=\"{cx}\" y=\"{cy}\" fill=\"#f2f5f7\" font-size=\"11\" \
             text-anchor=\"middle\" dominant-baseline=\"central\">{mark}</text>"
        );
    }

    // The item tag sits below the arrow on a machine (which has room) and
    // replaces it on a 1×1 source or sink (which does not).
    let tag = item_tag(cell.item);
    if !tag.is_empty() {
        let ty = if fh > 1 { cy + TILE / 2 } else { cy };
        let _ = write!(
            s,
            "<text x=\"{cx}\" y=\"{ty}\" fill=\"#e8edf2\" font-size=\"8\" \
             text-anchor=\"middle\" dominant-baseline=\"central\">{tag}</text>"
        );
    }

    s.push_str("</g>");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory_gen::{generate, LessonKind};
    use crate::world::Cell;

    fn assembler_grid() -> Grid {
        let mut g = Grid::new(5, 5);
        g.set(
            1,
            1,
            Cell {
                entity: Entity::Assembler,
                direction: Direction::East,
                item: Item::IronGear,
                ..Default::default()
            },
        );
        g
    }

    /// The whole reason this module exists: a 3×3 machine must be drawn as one
    /// 3×3 object. A viewer that drew it 1×1 would show a factory that imports
    /// as something else, which is the bug the footprint work just fixed.
    #[test]
    fn a_machine_is_drawn_across_every_tile_it_covers() {
        let svg = grid_to_svg(&assembler_grid());
        // Anchored at (1,1) with TILE=22: x=23, y=23, and 3 tiles wide minus
        // the 2-unit stroke inset.
        assert!(
            svg.contains("x=\"23\" y=\"23\" width=\"64\" height=\"64\""),
            "the assembler must span 3 tiles, not 1:\n{svg}"
        );
    }

    /// Body tiles are stored `Empty`, and a renderer that walked cells naively
    /// would draw eight blanks on top of the machine.
    #[test]
    fn a_machines_body_tiles_are_not_drawn_as_separate_entities() {
        let svg = grid_to_svg(&assembler_grid());
        assert_eq!(
            svg.matches("<g>").count(),
            1,
            "one machine must produce exactly one drawn object:\n{svg}"
        );
    }

    /// Hovering a body tile should name the machine standing on it, not report
    /// the `Empty` that is literally stored there.
    #[test]
    fn a_body_tile_reports_the_machine_that_covers_it() {
        let g = assembler_grid();
        assert!(g.get(2, 2).is_empty(), "(2,2) is a body tile");
        assert_eq!(
            describe(&g, 2, 2),
            "(2, 2) covered by the Assembler at (1, 1)"
        );
    }

    /// A splitter is 2×1 and rotates; drawing it square would misreport a real
    /// prototype's shape.
    #[test]
    fn a_rotated_splitter_is_drawn_tall_rather_than_wide() {
        let mut g = Grid::new(4, 4);
        g.set(
            1,
            1,
            Cell {
                entity: Entity::Splitter,
                direction: Direction::East,
                ..Default::default()
            },
        );
        let svg = grid_to_svg(&g);
        // 2×1 facing east becomes 1 wide, 2 tall: 20 by 42 after the inset.
        assert!(
            svg.contains("width=\"20\" height=\"42\""),
            "an east-facing splitter must be 1x2:\n{svg}"
        );
    }

    /// The report embeds this straight into HTML, so an unescaped or unbalanced
    /// fragment would corrupt the page rather than merely look wrong.
    #[test]
    fn every_generated_factory_renders_to_balanced_svg() {
        let mut drawn = 0;
        for &kind in LessonKind::all() {
            for seed in 0..8 {
                let Some(sample) = generate(kind, 11, seed) else {
                    continue;
                };
                drawn += 1;
                let svg = grid_to_svg(&sample.solution);
                assert!(svg.starts_with("<svg"), "{}: no svg root", kind.name());
                assert!(svg.ends_with("</svg>"), "{}: unclosed svg", kind.name());
                assert_eq!(
                    svg.matches("<g>").count(),
                    svg.matches("</g>").count(),
                    "{} seed {seed}: unbalanced groups",
                    kind.name()
                );
            }
        }
        // The footprint sweep in tests/blueprint_export.rs passed on an empty
        // set once already; this loop is the same shape, so it gets the same
        // guard.
        assert!(drawn > 0, "no factories were rendered at all");
    }
}
