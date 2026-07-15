//! Show one task that has several valid answers, and what the simulator makes
//! of them.
//!
//! `task_space` measures ambiguity as a number. This renders it: it finds a
//! single `ASSEMBLER_BANK` task that the generator answered more than one way,
//! and prints every answer next to the rate it delivers. That the answers differ
//! is what gives Best-of-N a choice; that their *rates* differ is what gives the
//! choice a right answer.
//!
//! Run: `cargo run --release --example ambiguity_demo`

use std::collections::BTreeMap;

use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::textual::render;
use diffusion_factorio::throughput;
use diffusion_factorio::world::{Entity, Grid};

const SIZE: usize = 11;
const SEEDS: u64 = 20_000;

/// Everything the model is conditioned on, and nothing else. Must key on all
/// four channels: [`render`] draws every sink as `K` whatever its recipe, so
/// keying on the picture would silently merge three different tasks into one.
/// Mirrors `conditioning_key` in `task_space`.
fn task_key(g: &Grid, observed: &[bool]) -> String {
    let mut s = String::new();
    for (i, &obs) in observed.iter().enumerate() {
        if obs {
            let c = g.cells[i];
            s.push_str(&format!(
                "{i}:{}:{}:{}:{};",
                c.entity as u8, c.direction as u8, c.item as u8, c.misc as u8
            ));
        }
    }
    for y in 0..g.height {
        for x in 0..g.width {
            if g.is_obstacle(x, y) {
                s.push_str(&format!("#{x},{y};"));
            }
        }
    }
    s
}

fn main() {
    // Two samples sharing a conditioning are two answers to the same question.
    let mut by_task: BTreeMap<String, (Grid, BTreeMap<String, f64>)> = BTreeMap::new();
    for seed in 0..SEEDS {
        let Some(sample) = generate(LessonKind::AssemblerBank, SIZE, seed) else {
            continue;
        };
        let (scaffold, observed) = sample.blank_to_scaffold();
        by_task
            .entry(task_key(&sample.solution, &observed))
            .or_insert_with(|| (scaffold, BTreeMap::new()))
            .1
            .insert(
                render(&sample.solution),
                throughput::score(&sample.solution),
            );
    }

    let (_, (scaffold, answers)) = by_task
        .iter()
        .max_by_key(|(_, (_, answers))| answers.len())
        .expect("the generator produces banks at this size");
    let recipe = scaffold
        .cells
        .iter()
        .find(|c| c.entity == Entity::Sink)
        .expect("every bank has a sink")
        .item;

    println!(
        "=== the task the model is given (the sink asks for {recipe:?}) ===\n{}",
        render(scaffold)
    );
    println!("=== {} valid answers to it ===\n", answers.len());
    for (answer, rate) in answers {
        println!("-- delivers {rate:.3} {recipe:?}/s --\n{answer}");
    }

    let rates: Vec<f64> = answers.values().copied().collect();
    let (lo, hi) = (
        rates.iter().copied().fold(f64::MAX, f64::min),
        rates.iter().copied().fold(0.0, f64::max),
    );
    let ambiguous = by_task.values().filter(|(_, a)| a.len() > 1).count();
    println!(
        "Same sources, same sink, same recipe: {} answers spanning {lo:.3}..{hi:.3}/s ({:.1}x). \
         Over {SEEDS} seeds, {ambiguous} of {} tasks admit more than one answer.",
        answers.len(),
        hi / lo,
        by_task.len(),
    );
}
