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
