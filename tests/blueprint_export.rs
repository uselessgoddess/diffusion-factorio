use base64::Engine;
use diffusion_factorio::blueprint::{blueprint_json, blueprint_string, grid_to_blueprint};
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
