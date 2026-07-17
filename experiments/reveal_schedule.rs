//! Why the factory appears in the last few frames, and what `steps` actually buys.
//!
//! The issue reports two things that look like separate observations:
//!
//! > Модель генерирует всю схему за последние 2-3 шага (по крайней мере анимация
//! > так выглядит, но увеличение шагов до 24 увеличивает качество).
//!
//! They are one fact. The sampler commits cells on a cosine schedule
//! ([`still_masked_after`]), and the *rate* it commits them at is that curve's
//! derivative — near zero at the start, steepest at the very end. The animation
//! is not misleading and the model is not procrastinating: the schedule is what
//! holds the grid back and then dumps it.
//!
//! No model is loaded here. The schedule is arithmetic over one integer, so the
//! numbers below are exact and this file calls the same function the sampler
//! does — an experiment that re-derived the curve could drift away from it.
//!
//! Run: `cargo run --release --example reveal_schedule`

use diffusion_factorio::sample::still_masked_after;

/// The grid from the issue's inference session: 13 wide, 9 tall, a source and a
/// sink pinned, everything else for the model to decide.
const WIDTH: usize = 13;
const HEIGHT: usize = 9;
const PINS: usize = 2;

/// `steps` values worth comparing: the sampler's default, the value the issue
/// found helped, and enough neighbours to see the trend.
const STEPS: [usize; 6] = [4, 8, 12, 16, 24, 32];

/// Replay the sampler's per-round reveal arithmetic exactly as `sample.rs` runs
/// it, and return how many cells each round commits.
fn reveals(steps: usize, masked0: usize) -> Vec<usize> {
    let mut remaining = masked0;
    (0..steps)
        .map(|step| {
            let reveal = remaining
                .saturating_sub(still_masked_after(step + 1, steps, masked0))
                .max(1)
                .min(remaining);
            remaining -= reveal;
            reveal
        })
        .collect()
}

fn main() {
    let masked0 = WIDTH * HEIGHT - PINS;
    println!(
        "A {WIDTH}x{HEIGHT} task with {PINS} pins: {masked0} cells for the model to commit.\n"
    );

    println!("Cells committed per round (the sampler's own arithmetic):");
    for steps in STEPS {
        let per_round = reveals(steps, masked0);
        let bars: Vec<String> = per_round.iter().map(|n| n.to_string()).collect();
        println!("  steps={steps:<3} {}", bars.join(" "));
    }

    println!("\nWhere the grid actually gets decided:");
    println!("  steps  last round   last 3 rounds   rounds to decide half");
    for steps in STEPS {
        let per_round = reveals(steps, masked0);
        let share = |cells: usize| 100.0 * cells as f64 / masked0 as f64;
        let last = per_round[steps - 1];
        let last3: usize = per_round.iter().rev().take(3).sum();
        // The first round after which fewer than half the cells are still masked.
        let half = (1..=steps)
            .find(|&k| still_masked_after(k, steps, masked0) * 2 <= masked0)
            .unwrap_or(steps);
        println!(
            "  {steps:<5}  {last:>3} ({:>4.1}%)  {last3:>4} ({:>4.1}%)     {half}/{steps} = {:.0}%",
            share(last),
            share(last3),
            100.0 * half as f64 / steps as f64,
        );
    }

    println!(
        "\nThe last column is the shape, and it barely moves: ~70% of the rounds\n\
         pass before half the grid is decided, at every setting. That is the curve,\n\
         not a tuning artefact — cos hits 0.5 at exactly 2/3 of the way along,\n\
         whatever you divide the way into, and the column reads a little higher only\n\
         because it counts whole rounds and rounds the crossing up.\n\
         \n\
         So `steps` does not change *when* the model commits. It changes how big the\n\
         final irreversible chunk is: the last round commits sin(pi/2/steps) of the\n\
         grid in one shot, with no round left to react to what it just did. That is\n\
         13% of this grid at steps=12 and 7% at steps=24, and it is what the issue\n\
         bought by raising steps. It costs one forward pass per step per candidate --\n\
         steps=24 with candidates=8 is 192 passes against the default's 96.\n\
         \n\
         The 2-3 frames the animation appears in are real, and the schedule put them\n\
         there: at steps=12 the last three rounds commit 38% of the grid between\n\
         them, at steps=24, 19%.\n\
         \n\
         (Percentages are what the sampler's integer arithmetic actually commits, so\n\
         they sit within a cell or two of the continuous curve: the loop reveals at\n\
         least one cell per round even where cos asks for none, which is why early\n\
         rounds at steps=32 read 1 rather than 0.)"
    );
}
