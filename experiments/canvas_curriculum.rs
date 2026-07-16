//! Cheap padding, or generation that knows the canvas has two sides?
//!
//! `experiments/grid_shape` established the cost of a square-only curriculum: the
//! identical task drops from 3.118 items/s on the trained 11x11 to 0.478 on the
//! 13x9 the issue infers on. That leaves the question of what to do about it, and
//! there were two candidates:
//!
//! * **Padding.** Keep generating squares, drop each one into the rectangle at a
//!   random offset, and leave the rest empty. Nothing in `factory_gen` changes.
//! * **Native.** Give every generator the canvas as width and height and let it
//!   use both. Every generator changes.
//!
//! Padding is obviously cheaper to write, so the only honest way to choose is to
//! measure what it costs. This experiment needs no model and no checkpoint: both
//! options are decided by the curriculum alone, so both can be measured before a
//! single step of training is paid for.
//!
//! ```text
//! cargo run --release --example canvas_curriculum
//! ```

use std::time::Instant;

use diffusion_factorio::factory_gen::{generate, Canvas, LessonKind};

/// Seeds per family per canvas. Generation is generate-and-verify, so a family
/// that "fits" can still fail; this is enough to tell "rare" from "never".
const SEEDS: u64 = 400;

/// The canvases to weigh the two options on. 11x11 is what the reported runs
/// trained on and 13x9 is what they inferred on; the rest are the pool's corners.
const CANVASES: [(Canvas, &str); 6] = [
    (Canvas::new(11, 11), "trained shape"),
    (Canvas::new(13, 9), "the issue's shape"),
    (Canvas::new(9, 13), "the same, turned"),
    (Canvas::new(9, 9), "pool floor"),
    (Canvas::new(15, 15), "pool ceiling"),
    (Canvas::new(15, 9), "widest gap"),
];

/// The largest square that can be padded into `canvas`.
///
/// This is the whole of the padding option, in one line. A square of side `s`
/// dropped anywhere inside `w` x `h` needs `s <= min(w, h)`: the short side is
/// the binding constraint, and everything past it on the long side is pad.
fn padded_square(canvas: Canvas) -> Canvas {
    Canvas::square(canvas.width.min(canvas.height))
}

/// Families that generate at least once in `SEEDS` tries on `canvas`.
///
/// Deliberately *not* `LessonKind::fits`: `fits` is arithmetic and this is the
/// claim that matters — a lesson the curriculum can actually put in front of the
/// model.
fn buildable(canvas: Canvas) -> Vec<LessonKind> {
    LessonKind::all()
        .iter()
        .copied()
        .filter(|&kind| (0..SEEDS).any(|seed| generate(kind, canvas, seed).is_some()))
        .collect()
}

fn names(kinds: &[LessonKind]) -> String {
    if kinds.is_empty() {
        return "-".to_owned();
    }
    kinds
        .iter()
        .map(|k| k.name())
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let all = LessonKind::all().len();

    println!("=== 1. What can each option teach, per canvas? ===");
    println!(
        "(a family counts only if it actually generated in {SEEDS} seeds --\n\
         'padded' generates the largest square the canvas admits and pads the rest)\n"
    );
    println!("      canvas  padded  native  lost to padding");
    for (canvas, note) in CANVASES {
        let native = buildable(canvas);
        let padded = buildable(padded_square(canvas));
        let lost: Vec<LessonKind> = native
            .iter()
            .copied()
            .filter(|k| !padded.contains(k))
            .collect();
        println!(
            "  {:>11}  {:>4}    {:>4}    {:<28}  {note}",
            canvas.to_string(),
            format!("{}/{all}", padded.len()),
            format!("{}/{all}", native.len()),
            names(&lost),
        );
    }

    println!("\n=== 2. What does a padded canvas spend on nothing? ===");
    println!("(cells the label can never fill, because no lesson is generated there)\n");
    println!("      canvas   cells  square  pad cells   dead");
    for (canvas, _) in CANVASES {
        let square = padded_square(canvas);
        let pad = canvas.area() - square.area();
        println!(
            "  {:>11}  {:>5}  {:>6}  {:>9}  {:>4.0}%",
            canvas.to_string(),
            canvas.area(),
            square.to_string(),
            pad,
            100.0 * pad as f64 / canvas.area() as f64,
        );
    }

    println!("\n=== 3. What does native generation cost to run? ===");
    println!("(the cheapness of padding is its only argument, so price the alternative)\n");
    println!("      canvas   samples/s   per sample");
    for (canvas, _) in CANVASES {
        let kinds = LessonKind::all()
            .iter()
            .copied()
            .filter(|k| k.fits(canvas))
            .collect::<Vec<_>>();
        let t0 = Instant::now();
        let mut built = 0usize;
        for seed in 0..SEEDS {
            let kind = kinds[(seed as usize) % kinds.len()];
            if generate(kind, canvas, seed).is_some() {
                built += 1;
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "  {:>11}   {:>9.0}   {:>7.3} ms",
            canvas.to_string(),
            built as f64 / secs,
            1000.0 * secs / built.max(1) as f64,
        );
    }

    println!(
        "\nRead section 1 first: padding cannot show the model a lesson wider than the\n\
         canvas is tall, and the two families that compose several machines into one\n\
         factory -- CIRCUIT_LINE at 11x5 and SHARED_LINE at 11x7 -- are both wider\n\
         than they are tall. On the issue's 13x9 they fit natively with room over and\n\
         padding drops them, which is to say padding drops exactly the lessons the\n\
         'invent new solutions' goal rests on.\n\n\
         The 9x13 row is where native generation is honest about its own limit: it\n\
         scores 6/8 too. Every family is templated in one orientation, so a lesson\n\
         11 cells wide does not fit a 9-wide canvas however tall it is. Rotating the\n\
         templates would buy that row back and is the obvious next lesson-side move;\n\
         it is not what the issue's 13x9 needs, and it is not what padding buys.\n\n\
         Section 2 is the second cost: the pad region is empty in every label, so it is\n\
         a free cue that the answer lives inside a square somewhere -- the opposite of\n\
         the habit we want.\n\n\
         Section 3 is what the choice cost. Generation happens on the CPU while the\n\
         backward pass is the run's clock, and at these rates it is not the constraint."
    );
}
