//! Export the model's categorical grid as an importable Factorio 2.x blueprint.
//!
//! Factorio blueprint strings are `0` followed by base64-encoded zlib JSON.
//! Source and sink cells are abstract environment anchors, so they are rendered
//! as constant-combinator markers carrying role/item tags. All placeable model
//! entities map to their vanilla Factorio prototype names.

use std::io::Write;

use anyhow::{bail, Context, Result};
use base64::Engine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::world::{Direction, Entity, Grid, Item, Misc};

/// Factorio 2.0 version stamp used by the reference project.
pub const FACTORIO_2_VERSION: u64 = 562_949_958_402_048;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlueprintEnvelope {
    pub blueprint: Blueprint,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Blueprint {
    pub item: String,
    pub label: String,
    pub entities: Vec<BlueprintEntity>,
    pub version: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlueprintEntity {
    pub entity_number: usize,
    pub name: String,
    pub position: Position,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<u8>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub belt_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_behavior: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Convert every non-empty cell to a vanilla Factorio blueprint entity.
pub fn grid_to_blueprint(grid: &Grid, label: impl Into<String>) -> Result<BlueprintEnvelope> {
    let mut entities = Vec::new();

    for y in 0..grid.height {
        for x in 0..grid.width {
            let cell = grid.get(x, y);
            if cell.entity == Entity::Empty {
                continue;
            }
            if !cell.is_consistent() {
                bail!("cannot export inconsistent cell at ({x}, {y}): {:?}", cell);
            }

            let position = entity_position(cell.entity, cell.direction, x, y);
            let direction = factorio_direction(cell.direction);
            let (name, belt_type, recipe, tags, control_behavior) = match cell.entity {
                Entity::Empty => unreachable!(),
                Entity::Source => marker_fields("source", cell.item),
                Entity::Sink => marker_fields("sink", cell.item),
                Entity::TransportBelt => fields("transport-belt"),
                Entity::UndergroundBelt => {
                    let belt_type = match cell.misc {
                        Misc::UndergroundDown => Some("input".to_owned()),
                        Misc::UndergroundUp => Some("output".to_owned()),
                        Misc::None => None,
                    };
                    ("underground-belt".to_owned(), belt_type, None, None, None)
                }
                Entity::Splitter => fields("splitter"),
                Entity::Inserter => fields("inserter"),
                Entity::Assembler => (
                    "assembling-machine-1".to_owned(),
                    None,
                    recipe_name(cell.item).map(str::to_owned),
                    None,
                    None,
                ),
            };

            entities.push(BlueprintEntity {
                entity_number: entities.len() + 1,
                name,
                position,
                direction: if matches!(cell.entity, Entity::Source | Entity::Sink) {
                    Some(4)
                } else {
                    direction
                },
                belt_type,
                recipe,
                tags,
                control_behavior,
            });
        }
    }

    Ok(BlueprintEnvelope {
        blueprint: Blueprint {
            item: "blueprint".to_owned(),
            label: label.into(),
            entities,
            version: FACTORIO_2_VERSION,
        },
    })
}

/// Compact JSON accepted by Factorio 2.x and useful for debugging exporters.
pub fn blueprint_json(blueprint: &BlueprintEnvelope) -> Result<String> {
    serde_json::to_string(blueprint).context("serialize Factorio blueprint JSON")
}

/// Encode the classic Factorio blueprint exchange string (`0` + base64(zlib)).
pub fn blueprint_string(blueprint: &BlueprintEnvelope) -> Result<String> {
    let json = blueprint_json(blueprint)?;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(json.as_bytes())
        .context("compress Factorio blueprint JSON")?;
    let compressed = encoder.finish().context("finish blueprint compression")?;
    let payload = base64::engine::general_purpose::STANDARD.encode(compressed);
    Ok(format!("0{payload}"))
}

fn fields(
    name: &str,
) -> (
    String,
    Option<String>,
    Option<String>,
    Option<serde_json::Value>,
    Option<serde_json::Value>,
) {
    (name.to_owned(), None, None, None, None)
}

fn marker_fields(
    role: &str,
    item: Item,
) -> (
    String,
    Option<String>,
    Option<String>,
    Option<serde_json::Value>,
    Option<serde_json::Value>,
) {
    let item_name = item_name(item);
    let role_signal = if role == "source" {
        "signal-output"
    } else {
        "signal-input"
    };
    let mut filters = vec![json!({
        "index": 2,
        "type": "virtual",
        "name": role_signal,
        "quality": "normal",
        "comparator": "=",
        "count": 1
    })];
    if let Some(name) = item_name {
        filters.insert(
            0,
            json!({
                "index": 1,
                "name": name,
                "quality": "normal",
                "comparator": "=",
                "count": 1
            }),
        );
    }
    (
        "constant-combinator".to_owned(),
        None,
        None,
        Some(json!({"diffusion-factorio-role": role, "item": item_name})),
        Some(json!({"sections": {"sections": [{"index": 1, "filters": filters}]}})),
    )
}

fn factorio_direction(direction: Direction) -> Option<u8> {
    match direction {
        Direction::None => None,
        Direction::North => Some(0),
        Direction::East => Some(4),
        Direction::South => Some(8),
        Direction::West => Some(12),
    }
}

fn entity_position(entity: Entity, direction: Direction, x: usize, y: usize) -> Position {
    let (ox, oy) = match entity {
        Entity::Assembler => (1.5, 1.5),
        Entity::Splitter if matches!(direction, Direction::North | Direction::South) => (1.0, 0.5),
        Entity::Splitter => (0.5, 1.0),
        _ => (0.5, 0.5),
    };
    Position {
        x: x as f64 + ox,
        y: y as f64 + oy,
    }
}

fn item_name(item: Item) -> Option<&'static str> {
    match item {
        Item::None => None,
        Item::IronPlate => Some("iron-plate"),
        Item::CopperPlate => Some("copper-plate"),
        Item::IronGear => Some("iron-gear-wheel"),
        Item::CopperCable => Some("copper-cable"),
        Item::GreenCircuit => Some("electronic-circuit"),
    }
}

fn recipe_name(item: Item) -> Option<&'static str> {
    item_name(item)
}
