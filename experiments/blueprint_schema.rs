//! Emit a real blueprint string and check it against the schema rules that
//! <https://fbe.teoxoy.com/> enforces.
//!
//! Background: an export imported into Factorio fine but fbe rejected it with
//!   instancePath: "/blueprint", keyword: "required",
//!   message: "must have required property 'icons'",
//!   params: { missingProperty: "icons" }, schema: ["version", "item", "icons"]
//! Factorio derives the library thumbnail from the first entity when `icons` is
//! absent, so it never complained — which is why this only surfaced in a strict
//! validator.
//!
//! Run: `cargo run --release --example blueprint_schema`
//! Then paste the printed string into https://fbe.teoxoy.com/ to confirm.

use diffusion_factorio::blueprint::{blueprint_json, blueprint_string, grid_to_blueprint};
use diffusion_factorio::factory_gen::{generate, Canvas, LessonKind};
use diffusion_factorio::textual::render;

/// The three properties the blueprint schema lists as required.
const REQUIRED: [&str; 3] = ["version", "item", "icons"];

fn main() -> anyhow::Result<()> {
    for &kind in LessonKind::all() {
        let Some(sample) = generate(kind, Canvas::square(11), 7) else {
            continue;
        };
        let envelope = grid_to_blueprint(&sample.solution, format!("{} lesson", kind.name()))?;
        let value: serde_json::Value = serde_json::from_str(&blueprint_json(&envelope)?)?;
        let bp = &value["blueprint"];

        println!("=== {} ===", kind.name());
        print!("{}", render(&sample.solution));

        for key in REQUIRED {
            let ok = bp.get(key).is_some();
            println!(
                "  {} required property {key:<8} {}",
                if ok { "ok  " } else { "FAIL" },
                bp.get(key)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "<missing>".into()),
            );
            anyhow::ensure!(ok, "missing required property '{key}'");
        }

        // The constant-combinators are the abstract source/sink anchors. They
        // are deliberate: Factorio has no "infinite source" entity, so the
        // markers carry the role/item tags that describe the environment.
        let combinators = bp["entities"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["name"] == "constant-combinator")
            .count();
        println!("  {combinators} constant-combinator marker(s) = source/sink anchors (by design)");
        println!("  blueprint string:\n{}\n", blueprint_string(&envelope)?);
    }

    println!("All lessons export with every schema-required property present.");
    Ok(())
}
