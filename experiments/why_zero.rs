//! Find factories the simulator calls functional but the grader scores 0, and
//! print enough of one to see why.
//!
//! [`sim::item_reaches_sink`] and [`throughput::score`] answer different
//! questions and do not have to agree: the first asks whether items *can* reach
//! the sink, the second asks how many arrive per second. A generator bug shows
//! up as a factory that passes the first and fails the second, and the ASCII
//! render alone is not enough to tell you which cell is at fault -- it draws an
//! inserter as `i` whichever way the hand swings, and a machine's body tiles as
//! `a` whether they are claimed or free.
//!
//! So this dumps the cells themselves. Run with a lesson name to narrow it:
//!
//! ```text
//! cargo run --release --example why_zero -- ASSEMBLER_CHAOS
//! ```

use diffusion_factorio::{
    factory_gen::{generate, LessonKind},
    sim, textual,
    throughput::throughput,
    world::{Entity, Grid},
};

/// Every non-empty cell, spelled out: what stands there, which way it faces and
/// what it carries. This is the view the glyph render cannot give you.
fn dump_cells(grid: &Grid) -> String {
    let mut out = String::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let c = grid.get(x, y);
            if c.entity == Entity::Empty {
                // A body tile of a machine reads as `Empty` but is claimed, and
                // that distinction is exactly the sort of thing being hunted
                // here -- so say so rather than skipping the cell.
                if let Some(a) = grid.anchor_at(x, y) {
                    out.push_str(&format!("  ({x},{y}) body of the machine at {a:?}\n"));
                } else if grid.is_obstacle(x, y) {
                    out.push_str(&format!("  ({x},{y}) obstacle\n"));
                }
                continue;
            }
            out.push_str(&format!(
                "  ({x},{y}) {:?} facing {:?} item {:?}\n",
                c.entity, c.direction, c.item
            ));
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let wanted = args.first().cloned();

    let mut shown = 0usize;
    for &kind in LessonKind::all() {
        if let Some(w) = &wanted {
            if kind.name() != w {
                continue;
            }
        }
        let mut zeros = 0usize;
        let mut total = 0usize;
        for seed in 0..200u64 {
            let Some(s) = generate(kind, 11, seed) else {
                continue;
            };
            total += 1;
            let report = throughput(&s.solution);
            if report.score > 0.0 || !sim::item_reaches_sink(&s.solution) {
                continue;
            }
            zeros += 1;
            // One worked example is worth more than fifty renders. The rest are
            // only counted, so the summary line says whether this is a corner
            // case or the common path.
            if zeros == 1 && shown < 3 {
                shown += 1;
                println!("=== {} seed {seed}: functional, scores 0 ===", kind.name());
                print!("{}", textual::render(&s.solution));
                println!("cells:");
                print!("{}", dump_cells(&s.solution));
                println!("deliveries:");
                for d in &report.deliveries {
                    println!("  {d:?}");
                }
            }
        }
        println!(
            "{}: {zeros} of {total} generated factories are functional but score 0",
            kind.name()
        );
    }
}
