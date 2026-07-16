//! Look at the training data.
//!
//! Issue #9: *"я так как не особо вижу что там обучается"* — "I can't really
//! see what's being trained". Before asking whether the model's factories are
//! any good, you have to be able to see the factories it is being *shown*.
//! Every lesson family, several seeds each, drawn at Factorio's own footprints,
//! with the simulator's verdict and a paste-ready blueprint string per factory.
//!
//! ```text
//! cargo run --release --example gallery
//! ```
//!
//! Writes `gallery.html`: no network, no build step, open it in a browser.
//! The blueprint string under each factory is the escape hatch — paste it into
//! the real game and the picture stops being our opinion of the layout.

use diffusion_factorio::blueprint::{blueprint_string, grid_to_blueprint};
use diffusion_factorio::factory_gen::{generate, Canvas, LessonKind};
use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::throughput::throughput;
use diffusion_factorio::viewer::grid_to_svg;
use std::fmt::Write as _;
use std::fs;

/// Grid side. Matches the size the 5,000-step run trained at, so what you see
/// here is what the model saw.
const SIZE: usize = 11;
/// Seeds per lesson family.
const SEEDS: u64 = 6;

fn main() -> anyhow::Result<()> {
    let mut cards = String::new();
    let (mut total, mut routed) = (0usize, 0usize);

    for &kind in LessonKind::all() {
        let mut tiles = String::new();
        let mut drawn = 0;
        for seed in 0..SEEDS {
            let Some(sample) = generate(kind, Canvas::square(SIZE), seed) else {
                continue;
            };
            drawn += 1;
            total += 1;

            let grid = &sample.solution;
            let works = item_reaches_sink(grid);
            routed += usize::from(works);
            let report = throughput(grid);

            // If a factory cannot be exported, that is the interesting fact
            // about it, so say so rather than skipping it.
            let bp = grid_to_blueprint(grid, format!("{} #{seed}", kind.name()))
                .and_then(|b| blueprint_string(&b))
                .unwrap_or_else(|e| format!("export failed: {e}"));

            let _ = write!(
                tiles,
                "<figure><figcaption>seed {seed} · \
                 <span class=\"{}\">{}</span> · {:.3}/s</figcaption>{}\
                 <textarea readonly rows=\"2\" spellcheck=\"false\">{bp}</textarea></figure>",
                if works { "ok" } else { "bad" },
                if works { "routes" } else { "dead" },
                report.score,
                grid_to_svg(grid),
            );
        }
        let _ = write!(
            cards,
            "<section><h2>{} <small>{drawn} of {SEEDS} seeds produced a factory{}</small></h2>\
             <div class=\"row\">{tiles}</div></section>",
            kind.name(),
            if kind.is_ambiguous() {
                " · more than one correct answer"
            } else {
                ""
            },
        );
    }

    let html = PAGE.replace("__CARDS__", &cards).replace(
        "__SUMMARY__",
        &format!("{routed} of {total} route end to end"),
    );
    fs::write("gallery.html", &html)?;
    println!("gallery.html: {total} factories, {routed} route end to end");
    Ok(())
}

const PAGE: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>diffusion-factorio lesson gallery</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#101418;color:#e8edf2;font:14px/1.45 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{margin:0}
.intro{color:#9eabb6;margin:5px 0 22px}
section{background:#182027;border:1px solid #293740;border-radius:10px;padding:16px;margin:14px 0}
h2{font-size:17px;margin:0 0 12px}h2 small{color:#9eabb6;font-weight:400;font-size:12px}
.row{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:0;width:260px}
figcaption{font-size:12px;color:#aeb9c2;margin-bottom:6px}
.ok{color:#5ee6a8}.bad{color:#ff7b9c}
svg.factory{display:block;width:260px;height:auto;border-radius:5px}
textarea{width:100%;margin-top:6px;background:#0b0e11;color:#6f7d88;border:1px solid #293740;
 border-radius:4px;font:10px/1.3 ui-monospace,monospace;resize:vertical;padding:4px}
</style></head><body><main>
<h1>Lesson gallery</h1>
<div class="intro">What the model is trained on, drawn at the footprints Factorio enforces.
Throughput is the simulator's grade. Each blueprint string pastes into the real game — that is
the only check that does not take our own simulator's word for it. · __SUMMARY__</div>
__CARDS__
</main></body></html>"#;
