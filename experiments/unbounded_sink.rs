//! Can a *lesson* press a source against a sink?
//!
//! [`throughput`] used to report such a sink at `inf` items/second: a source is
//! an unlimited well, a sink is never the constraint, and with nothing built in
//! between there is nothing to cap the flow. `serde_json` writes a non-finite
//! float as `null`, so the first hand-painted task POSTed to `bin/serve` killed
//! the page on `null.toFixed`. It took a model's own garbage to find it.
//!
//! The fix reports that sink at the 0 it delivers, which changes the score of
//! any grid holding one. This asks how much of the repo's measured ground that
//! moves — every published throughput number is folded from lesson solutions, so
//! if no lesson contains the arrangement, no published number moves:
//!
//! ```text
//! cargo run --release --example unbounded_sink
//! ```

use diffusion_factorio::{
    factory_gen::{generate, LessonKind},
    throughput::throughput,
    world::{Entity, Grid, Item},
};

/// Sinks with a source pressed against them, and how many of those sources the
/// sink would actually *count*.
///
/// A source offers its item to its whole perimeter (`sim::flow_targets`) and a
/// sink accepts from anything (`throughput::accepts_from`), so pressing the two
/// together always builds the edge. Only the sink's filter decides whether the
/// unbounded flow lands in the number it reports: a gear sink sums gears, so an
/// unlimited well of *plate* against its wall is offered, ignored, and finite.
fn pressed(grid: &Grid) -> (usize, usize) {
    let (mut touching, mut counted) = (0, 0);
    for y in 0..grid.height {
        for x in 0..grid.width {
            let sink = grid.get(x, y);
            if sink.entity != Entity::Sink {
                continue;
            }
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if !grid.in_bounds(nx, ny) {
                    continue;
                }
                let neighbour = grid.get(nx as usize, ny as usize);
                if neighbour.entity != Entity::Source {
                    continue;
                }
                touching += 1;
                // `sim::sink_accepts`, which is crate-private: an unfiltered
                // sink takes everything, a filtered one takes only its item.
                if sink.item == Item::None || sink.item == neighbour.item {
                    counted += 1;
                }
            }
        }
    }
    (touching, counted)
}

fn main() {
    const PER_KIND: u64 = 200;
    println!("{PER_KIND} lessons per family, size 11\n");
    println!(
        "{:<20} {:>8} {:>9} {:>8} {:>11}",
        "family", "lessons", "touching", "counted", "non-finite"
    );

    let (mut lessons, mut touching, mut counted, mut nonfinite) = (0, 0, 0, 0);
    for &kind in LessonKind::all() {
        let (mut k_lessons, mut k_touching, mut k_counted, mut k_nonfinite) = (0, 0, 0, 0);
        for seed in 0..PER_KIND {
            let Some(sample) = generate(kind, 11, seed) else {
                continue;
            };
            k_lessons += 1;
            let (t, c) = pressed(&sample.solution);
            k_touching += t;
            k_counted += c;
            k_nonfinite += throughput(&sample.solution)
                .deliveries
                .iter()
                .filter(|d| !d.achieved.is_finite())
                .count();
        }
        println!(
            "{:<20} {k_lessons:>8} {k_touching:>9} {k_counted:>8} {k_nonfinite:>11}",
            format!("{kind:?}")
        );
        lessons += k_lessons;
        touching += k_touching;
        counted += k_counted;
        nonfinite += k_nonfinite;
    }

    println!(
        "\n{:<20} {lessons:>8} {touching:>9} {counted:>8} {nonfinite:>11}",
        "TOTAL"
    );
    println!(
        "\n{}",
        if nonfinite == 0 {
            "No lesson reports an unbounded rate, so no lesson's score moves: every\n\
             throughput number published from the curriculum is unchanged by reporting\n\
             an unbounded sink at the 0 it delivers.\n\n\
             Note what does the saving, because it is not the layout. The curriculum\n\
             does press a source against a sink, and a source offers its item to its\n\
             whole perimeter, so the edge gets built every time. What keeps the number\n\
             finite is that the two never agree on an item: a crafting lesson's source\n\
             carries the ingredient and its sink wants the product, so the well of plate\n\
             against the gear sink's wall is offered and ignored. The lessons that *do*\n\
             match items end to end are the moving ones, and those always run a belt in\n\
             between -- a belt caps at BELT_RATE, which is the whole reason the rate is\n\
             a number at all.\n\n\
             Both halves of that are properties of a generator that only ever emits\n\
             working factories. A hand-painted task owes us neither: it can ask for an\n\
             unfiltered sink, which counts every item, or drop a matching source on its\n\
             doorstep. That is how one click on the viewer found what 1,600 lessons\n\
             could not."
        } else {
            "A lesson DOES report an unbounded rate -- published throughput numbers\n\
             folded from the curriculum move, and want re-measuring."
        }
    );
}
