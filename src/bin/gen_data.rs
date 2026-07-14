//! Inspect the procedural training data: generate a few lessons and print each
//! as the full solution, the blanked (masked) view the model would see, and its
//! functional status. Pure CPU, no ML — handy for eyeballing the data pipeline.
//!
//! Usage: `cargo run --bin gen_data -- --size 11 --count 2`

use clap::Parser;
use diffusion_factorio::factory_gen::{generate, LessonKind};
use diffusion_factorio::sim::item_reaches_sink;
use diffusion_factorio::textual::render;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Parser)]
#[command(about = "Generate and display procedural factory lessons")]
struct Args {
    /// Grid side length.
    #[arg(long, default_value_t = 11)]
    size: usize,
    /// How many examples per lesson kind.
    #[arg(long, default_value_t = 1)]
    count: usize,
    /// Base RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let mut blank_rng = ChaCha8Rng::seed_from_u64(args.seed ^ 0xB1A2);

    for &kind in LessonKind::all() {
        for i in 0..args.count {
            let seed = args
                .seed
                .wrapping_add((kind as u64) << 32)
                .wrapping_add(i as u64);
            let Some(sample) = generate(kind, args.size, seed) else {
                println!(
                    "[{}] (could not generate at size {})\n",
                    kind.name(),
                    args.size
                );
                continue;
            };
            let (partial, _observed) = sample.blank(None, &mut blank_rng);
            println!("=== {} (seed {seed}) ===", kind.name());
            println!(
                "functional: {}  |  removable: {}  protected: {}",
                item_reaches_sink(&sample.solution),
                sample.removable.len(),
                sample.protected.len(),
            );
            println!("-- solution --\n{}", render(&sample.solution));
            println!(
                "-- masked input (what the model sees) --\n{}",
                render(&partial)
            );
        }
    }
}
