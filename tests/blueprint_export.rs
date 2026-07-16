use base64::Engine;
use diffusion_factorio::blueprint::{blueprint_json, blueprint_string, grid_to_blueprint};
use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::textual::render;
use diffusion_factorio::world::{Cell, Direction, Entity, Grid, Item, Misc};
use flate2::read::ZlibDecoder;
use std::io::Read;

fn example_grid() -> Grid {
    let mut grid = Grid::new(5, 1);
    grid.set(
        0,
        0,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    grid.set(1, 0, Cell::belt(Direction::East));
    grid.set(
        2,
        0,
        Cell {
            entity: Entity::UndergroundBelt,
            direction: Direction::East,
            misc: Misc::UndergroundDown,
            ..Default::default()
        },
    );
    grid.set(
        4,
        0,
        Cell {
            entity: Entity::Sink,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    grid
}

#[test]
fn grid_maps_to_factorio_entities_and_directions() {
    let blueprint = grid_to_blueprint(&example_grid(), "model reconstruction").unwrap();
    let entities = &blueprint.blueprint.entities;

    assert_eq!(entities.len(), 4);
    assert_eq!(entities[0].name, "constant-combinator");
    assert_eq!(entities[1].name, "transport-belt");
    assert_eq!(entities[1].direction, Some(4));
    assert_eq!(entities[2].name, "underground-belt");
    assert_eq!(entities[2].belt_type.as_deref(), Some("input"));
    assert_eq!(entities[3].name, "constant-combinator");
    assert_eq!(entities[0].entity_number, 1);
    assert_eq!(entities[3].entity_number, 4);
}

#[test]
fn blueprint_string_is_factorio_zlib_json_format() {
    let blueprint = grid_to_blueprint(&example_grid(), "round trip").unwrap();
    let json = blueprint_json(&blueprint).unwrap();
    let encoded = blueprint_string(&blueprint).unwrap();

    assert!(encoded.starts_with('0'));
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(&encoded[1..])
        .unwrap();
    let mut decoder = ZlibDecoder::new(compressed.as_slice());
    let mut decoded = String::new();
    decoder.read_to_string(&mut decoded).unwrap();

    assert_eq!(decoded, json);
    let value: serde_json::Value = serde_json::from_str(&decoded).unwrap();
    assert_eq!(value["blueprint"]["item"], "blueprint");
    assert_eq!(value["blueprint"]["label"], "round trip");
}

/// The blueprint schema requires `version`, `item` and `icons`. Factorio itself
/// imports a blueprint without `icons`, so this only shows up when a strict
/// validator (https://fbe.teoxoy.com/) rejects the export with
/// `must have required property 'icons'`.
#[test]
fn export_carries_every_schema_required_property() {
    let blueprint = grid_to_blueprint(&example_grid(), "schema").unwrap();
    let value: serde_json::Value =
        serde_json::from_str(&blueprint_json(&blueprint).unwrap()).unwrap();
    let bp = &value["blueprint"];

    for required in ["version", "item", "icons"] {
        assert!(
            bp.get(required).is_some(),
            "blueprint must have required property '{required}'"
        );
    }
    assert!(
        !bp["icons"].as_array().unwrap().is_empty(),
        "an empty icons array is as invalid as a missing one"
    );
}

#[test]
fn icons_describe_the_factory_and_are_indexed_from_one() {
    let blueprint = grid_to_blueprint(&example_grid(), "icons").unwrap();
    let icons = &blueprint.blueprint.icons;

    // Source and sink both carry IronPlate, so the icon list dedupes to one.
    assert_eq!(icons.len(), 1);
    assert_eq!(icons[0].signal.name, "iron-plate");
    assert_eq!(icons[0].signal.kind, "item");
    assert_eq!(icons[0].index, 1);
}

#[test]
fn icons_lead_with_the_product_then_the_input() {
    let mut grid = Grid::new(3, 1);
    grid.set(
        0,
        0,
        Cell {
            entity: Entity::Source,
            item: Item::CopperPlate,
            ..Default::default()
        },
    );
    grid.set(1, 0, Cell::belt(Direction::East));
    grid.set(
        2,
        0,
        Cell {
            entity: Entity::Sink,
            item: Item::CopperCable,
            ..Default::default()
        },
    );

    let blueprint = grid_to_blueprint(&grid, "product first").unwrap();
    let names: Vec<&str> = blueprint
        .blueprint
        .icons
        .iter()
        .map(|i| i.signal.name.as_str())
        .collect();
    assert_eq!(names, vec!["copper-cable", "copper-plate"]);
    assert_eq!(blueprint.blueprint.icons[1].index, 2);
}

/// A factory can legitimately have no named items; the schema still demands a
/// non-empty `icons`.
#[test]
fn untyped_factory_still_gets_a_valid_icon() {
    let mut grid = Grid::new(2, 1);
    grid.set(0, 0, Cell::belt(Direction::East));
    grid.set(1, 0, Cell::belt(Direction::East));

    let blueprint = grid_to_blueprint(&grid, "no items").unwrap();
    assert_eq!(blueprint.blueprint.icons.len(), 1);
    assert_eq!(blueprint.blueprint.icons[0].signal.name, "transport-belt");
}

/// Factorio renders at most four icons; a factory handling more items must not
/// produce an over-long list.
#[test]
fn icons_are_capped_at_four() {
    let mut grid = Grid::new(6, 2);
    for (x, item) in [
        Item::IronPlate,
        Item::CopperPlate,
        Item::IronGear,
        Item::CopperCable,
        Item::GreenCircuit,
    ]
    .into_iter()
    .enumerate()
    {
        grid.set(
            x,
            0,
            Cell {
                entity: Entity::Sink,
                item,
                ..Default::default()
            },
        );
    }

    let icons = grid_to_blueprint(&grid, "many items")
        .unwrap()
        .blueprint
        .icons;
    assert_eq!(icons.len(), 4);
    assert_eq!(
        icons.iter().map(|i| i.index).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

/// The footprint of the *vanilla prototype*, by the name and direction we
/// export — not by our own [`Entity::footprint`]. Deriving these from the world
/// model would make the test below circular: it would prove the exporter agrees
/// with us, when what matters is that it agrees with Factorio. These are the
/// sizes the game actually enforces on import.
///
/// Factorio rotates a footprint with the entity, so the direction is part of the
/// question and not an ornament: a `splitter` is two tiles wide and one tall
/// facing north or south, and one wide by two tall facing east or west. A square
/// prototype is unaffected, which is why an `assembling-machine-1` may ignore its
/// facing here.
///
/// This rule went untested until `SHARED_LINE` became the first family to place a
/// splitter at all. The sweep below had swept a splitter exactly zero times, so a
/// size table that could not rotate looked correct for as long as the vocabulary
/// had a hole in it.
fn prototype_size(name: &str, direction: Option<u8>) -> (f64, f64) {
    let (w, h) = match name {
        "assembling-machine-1" => (3.0, 3.0),
        "splitter" => (2.0, 1.0),
        "transport-belt" | "underground-belt" | "inserter" | "constant-combinator" => (1.0, 1.0),
        other => panic!("unknown prototype {other}: give it a size before exporting it"),
    };
    // 4 is east and 12 is west, in the game's sixteenths-of-a-turn encoding.
    match direction {
        Some(4) | Some(12) => (h, w),
        _ => (w, h),
    }
}

/// Every pair of exported entities whose vanilla footprints occupy the same
/// ground. Factorio refuses to import such a blueprint outright.
fn collisions(grid: &Grid) -> Vec<(String, String)> {
    let bp = grid_to_blueprint(grid, "overlap check").unwrap();
    let boxes: Vec<_> = bp
        .blueprint
        .entities
        .iter()
        .map(|e| {
            let (w, h) = prototype_size(&e.name, e.direction);
            let (x, y) = (e.position.x, e.position.y);
            (
                e.name.clone(),
                x - w / 2.0,
                y - h / 2.0,
                x + w / 2.0,
                y + h / 2.0,
            )
        })
        .collect();

    let mut found = Vec::new();
    for (i, a) in boxes.iter().enumerate() {
        for b in &boxes[i + 1..] {
            if a.1 < b.3 && b.1 < a.3 && a.2 < b.4 && b.2 < a.4 {
                found.push((a.0.clone(), b.0.clone()));
            }
        }
    }
    found
}

/// The bug this file now guards, reproduced by hand — and the proof that the
/// sweep below is looking for something real rather than passing on an empty
/// set.
///
/// This is the shape `gen_assembler_line` used to emit: `S i a i K` on one row,
/// with the assembler treated as a single cell. The world model said 1×1 and
/// nothing in the repo disagreed, but `blueprint.rs` has always exported a real
/// 3×3 `assembling-machine-1`, which swallows the two inserters standing beside
/// it. The simulator scored these factories as perfectly functional for the
/// entire life of the lesson; only Factorio could see the problem, and nothing
/// here ever asked Factorio.
#[test]
fn the_old_one_cell_assembler_layout_would_be_rejected_by_factorio() {
    let mut grid = Grid::new(5, 1);
    grid.set(
        0,
        0,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    for x in [1, 3] {
        grid.set(
            x,
            0,
            Cell {
                entity: Entity::Inserter,
                direction: Direction::East,
                ..Default::default()
            },
        );
    }
    grid.set(
        2,
        0,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::IronGear,
            ..Default::default()
        },
    );
    grid.set(
        4,
        0,
        Cell {
            entity: Entity::Sink,
            item: Item::IronGear,
            ..Default::default()
        },
    );

    assert!(
        !grid.is_consistent(),
        "the world model must now reject what it used to generate"
    );
    assert!(
        !collisions(&grid).is_empty(),
        "and the export must be seen to collide, or the sweep below proves nothing"
    );
}

/// No factory the curriculum produces may export to a blueprint Factorio would
/// reject. Sweeping every family means this covers any lesson added later,
/// whether or not its author thinks about footprints.
#[test]
fn no_generated_factory_exports_overlapping_entities() {
    for &kind in LessonKind::all() {
        let mut built = 0;
        for seed in 0..64 {
            let Some(sample) = generate(kind, 11, seed) else {
                continue;
            };
            built += 1;
            let found = collisions(&sample.solution);
            assert!(
                found.is_empty(),
                "{} seed {seed}: {found:?} overlap, so Factorio would reject \
                 this blueprint:\n{}",
                kind.name(),
                render(&sample.solution),
            );
        }
        // A family that generates nothing would sail through the loop above and
        // report success. That is exactly how the first draft of this test
        // fooled me: shrinking the assembler back to 1×1 left the inserters off
        // its perimeter, `item_reaches_sink` refused every layout, and the
        // "no collisions" result was over an empty set.
        assert!(built > 0, "{} generated no factories at all", kind.name());
    }
}

/// Factorio anchors an entity by its centre; we anchor it by its top-left tile.
/// A machine placed at the grid's own origin must therefore come out centred on
/// its footprint, which is the arithmetic the bug above got wrong in one file
/// and right in the other.
#[test]
fn an_entity_is_centred_on_the_tiles_it_covers() {
    let mut grid = Grid::new(4, 4);
    grid.set(
        0,
        0,
        Cell {
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::IronGear,
            ..Default::default()
        },
    );
    let bp = grid_to_blueprint(&grid, "centring").unwrap();
    let machine = &bp.blueprint.entities[0];
    assert_eq!(machine.name, "assembling-machine-1");
    // Anchored at (0,0), covering tiles 0..3 on both axes: the centre is 1.5.
    assert_eq!((machine.position.x, machine.position.y), (1.5, 1.5));
}

/// The assembler above cannot catch a footprint that fails to rotate, because
/// a 3×3 rotated is a 3×3. A splitter can: it is 2×1 lying north-south and 1×2
/// facing east, so its centre moves when it turns. Everything that reads a
/// blueprint — the game, and `prototype_size` in this file — has to turn with it.
///
/// Worth stating twice over because the export is the only artefact here that
/// leaves the repo. A wrong centre is not a wrong number in a metric; it is a
/// blueprint the player pastes and Factorio refuses.
#[test]
fn a_splitter_turns_its_footprint_when_it_turns() {
    for (direction, centre) in [
        // Lying north-south it covers tiles (0,0) and (1,0): two wide, one tall.
        (Direction::North, (1.0, 0.5)),
        // Facing east it covers (0,0) and (0,1) instead: one wide, two tall.
        (Direction::East, (0.5, 1.0)),
    ] {
        let mut grid = Grid::new(4, 4);
        grid.set(
            0,
            0,
            Cell {
                entity: Entity::Splitter,
                direction,
                ..Default::default()
            },
        );
        let bp = grid_to_blueprint(&grid, "splitter centring").unwrap();
        let splitter = &bp.blueprint.entities[0];
        assert_eq!(splitter.name, "splitter");
        assert_eq!(
            (splitter.position.x, splitter.position.y),
            centre,
            "a splitter facing {direction:?} is centred wrong"
        );
    }
}

/// An inserter's exported `direction` must name the tile it **drops into**.
///
/// This is the one convention in the format that is a coin-flip you cannot see
/// yourself losing: get it backwards and every inserter in every blueprint runs
/// the wrong way, the factory does nothing in-game, and — exactly as with the
/// footprint bug — our own simulator scores it as perfect, because our simulator
/// is reading our own convention back to us.
///
/// The reference states the rule from both sides and verified it in a live game
/// (`factorion-mod/server/blueprint.py:12`, "Inserters' blueprint direction
/// points to their *drop tile*, not pickup", and `factorion.py:587`, "Blueprint
/// direction = drop tile; model direction = pickup"). It has to flip by 8 on the
/// way out because its model points inserters at their *pickup*.
///
/// Ours points them at their drop already: `sim::flow_targets` pushes an
/// inserter's flow to `p + d`, and `throughput::accepts_from` has it pick up
/// from `p - d`. So our direction and Factorio's mean the same thing and we emit
/// it unflipped. This test is what makes that a decision rather than an
/// accident — if either convention is ever inverted, one side of it breaks here.
#[test]
fn an_inserter_points_at_the_tile_it_drops_into() {
    // S i K : the inserter takes from the source behind it and feeds the sink
    // in front of it.
    let mut grid = Grid::new(3, 1);
    grid.set(
        0,
        0,
        Cell {
            entity: Entity::Source,
            item: Item::IronPlate,
            ..Default::default()
        },
    );
    grid.set(
        1,
        0,
        Cell {
            entity: Entity::Inserter,
            direction: Direction::East,
            ..Default::default()
        },
    );
    grid.set(
        2,
        0,
        Cell {
            entity: Entity::Sink,
            item: Item::IronPlate,
            ..Default::default()
        },
    );

    // Our half of the claim: east means the plate lands to the east.
    assert!(
        diffusion_factorio::sim::item_reaches_sink(&grid),
        "our own model must agree the inserter feeds the sink to its east"
    );

    // Factorio's half: direction 4 is east, and Factorio reads it as the drop.
    let bp = grid_to_blueprint(&grid, "inserter facing").unwrap();
    let inserter = bp
        .blueprint
        .entities
        .iter()
        .find(|e| e.name == "inserter")
        .expect("the inserter must be exported");
    assert_eq!(
        inserter.direction,
        Some(4),
        "an east-dropping inserter must export as direction 4 (east), unflipped"
    );
}
