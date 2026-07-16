//! Best-of-N: draw several candidate factories for the same task and keep the
//! one the simulator likes best.
//!
//! This is the cheapest quality win available to us — no retraining, no new
//! loss, no gradient. The sampler is already stochastic at `temperature > 0`,
//! and [`crate::throughput`] can now rank two factories that both work, so the
//! only missing piece was to draw more than once and sort.
//!
//! It is also an honest probe of whether the model holds a *distribution* over
//! factories or a single memorised answer. [`BestOfN::distinct`] counts how many
//! different grids the draws produced: if it stays at 1, Best-of-N buys nothing
//! and no larger `n` will change that — the curriculum is the thing to fix.
//!
//! Verification is by simulation, not by the model's own confidence. A candidate
//! the denoiser is sure about can still be a factory that delivers nothing, and
//! the whole point is to let the simulator, not the network, have the last word.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use burn::prelude::*;

use crate::model::Denoiser;
use crate::sample::{reconstruct_with_diagnostics, ReconstructionDiagnostics, SampleConfig};
use crate::throughput;
use crate::world::Grid;

/// How to draw and rank the candidates.
#[derive(Clone, Debug)]
pub struct BestOfNConfig {
    /// Candidates per task. `1` degenerates to plain single-shot sampling.
    pub n: usize,
    /// Base sampler settings. `temperature` must be above zero or every draw is
    /// the same greedy argmax; `seed` is a base — draw `i` runs at `seed + i`.
    pub sample: SampleConfig,
    /// Break throughput ties by preferring the factory that uses fewer parts.
    ///
    /// The issue asks for a model "награждалась за компактность" — rewarded for
    /// compactness. Compactness cannot be an objective of its own: the most
    /// compact factory is the empty one, and it delivers nothing. So it never
    /// outranks throughput here, it only chooses among the candidates that
    /// already tie on it. See [`parts`] for what is counted.
    pub prefer_compact: bool,
}

impl Default for BestOfNConfig {
    fn default() -> Self {
        Self {
            n: 8,
            sample: SampleConfig {
                temperature: 1.0,
                ..Default::default()
            },
            prefer_compact: true,
        }
    }
}

/// How many parts a factory is built from: one per entity the grid holds.
///
/// Footprints are implied rather than stored — only an entity's anchor is a
/// cell — so this counts a 3×3 assembler once, as the one machine it is, and
/// not nine times. Two candidates for the same task need the same machines and
/// the same source and sink, so what this actually ranks is the belt and
/// inserter run between them: the part of the factory the model chose.
///
/// Obstacles are terrain rather than anything the model built, and they live on
/// their own plane, so they are never counted.
pub fn parts(grid: &Grid) -> usize {
    grid.cells.iter().filter(|c| !c.is_empty()).count()
}

/// What a draw actually delivers: its simulator score if it can be built, and
/// zero if it cannot.
///
/// [`throughput::score`] propagates flow through whatever the grid holds without
/// asking whether Factorio would accept it, so two assemblers sharing a tile can
/// score well and still be an import error. That number is a simulation of a
/// factory that does not exist. Nothing is delivered by a factory nobody can
/// build, so this reports nothing, and the raw figure stays available through
/// [`throughput::throughput`] for anyone diagnosing why.
pub fn usable_score(grid: &Grid) -> f64 {
    if grid.is_consistent() {
        throughput::score(grid)
    } else {
        0.0
    }
}

/// A draw, reduced to what ranking looks at.
#[derive(Clone, Copy, Debug)]
struct Ranked {
    /// Whether [`Grid::is_consistent`] holds: every cell well-formed and every
    /// footprint inside the grid, off the obstacles, and clear of every other.
    buildable: bool,
    /// [`usable_score`], not the raw simulator figure.
    score: f64,
    parts: usize,
}

/// Does a draw beat the best so far?
///
/// Buildable first, then throughput, then compactness among draws that already
/// tie on it.
///
/// **Buildable outranks throughput** because an unbuildable factory's throughput
/// is fiction. `throughput::score` propagates flow through whatever the grid
/// holds without asking whether Factorio would accept it, so two assemblers
/// sharing a tile can score well and still be an import error. Ranking on that
/// number is ranking on a simulation of a factory that cannot exist. This is
/// not the same guard as compactness below: a compact factory is real and merely
/// worth less, an unbuildable one is not there.
///
/// It also closes the path that returned the user an unimportable layout even
/// when a fine one was drawn. `best_score` starts at negative infinity, so draw
/// 0 always won it; when every draw delivered 0.0 items/s, `score > best` was
/// false for every later draw and `tied` was false because it requires
/// `best > 0.0`. Draw 0 therefore stood unconditionally, buildable or not, and
/// the slate behind it went unread.
///
/// Throughput still outranks compactness, and the `best > 0.0` is what keeps the
/// empty factory from winning: with every candidate at 0.0 the emptiest is the
/// most compact, and rewarding it would be rewarding the model for giving up.
fn beats(cand: Ranked, best: Ranked, prefer_compact: bool) -> bool {
    if cand.buildable != best.buildable {
        return cand.buildable;
    }
    let tied = cand.score == best.score && best.score > 0.0;
    cand.score > best.score || (prefer_compact && tied && cand.parts < best.parts)
}

/// What Best-of-N produced for one task.
#[derive(Clone, Debug)]
pub struct BestOfN {
    /// The winning candidate, diagnostics intact so it still renders in reports.
    pub best: ReconstructionDiagnostics,
    /// [`usable_score`] of every draw, in draw order.
    pub scores: Vec<f64>,
    /// [`parts`] of every draw, in draw order.
    pub parts: Vec<usize>,
    /// Whether each draw can be built, in draw order. A draw that cannot scores
    /// 0.0 above, and this is what tells that apart from a factory that was built
    /// correctly and simply delivers nothing.
    pub buildable: Vec<bool>,
    /// Distinct grids among the draws.
    pub distinct: usize,
}

impl BestOfN {
    /// Items/second delivered by the winner.
    ///
    /// The maximum is the winner's own score, not a figure read off some other
    /// draw: an unbuildable draw scores 0.0, so it can never hold a maximum that
    /// the buildable winner does not already match.
    pub fn best_score(&self) -> f64 {
        self.scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// What single-shot sampling would have returned: the first draw.
    pub fn first_score(&self) -> f64 {
        self.scores[0]
    }

    /// Items/second the extra draws bought. Never negative — the first draw is
    /// itself a candidate, so Best-of-N cannot do worse than sampling once.
    pub fn gain(&self) -> f64 {
        self.best_score() - self.first_score()
    }

    /// Parts the winner is built from.
    pub fn best_parts(&self) -> usize {
        parts(&self.best.grid)
    }

    /// Parts saved against the roomiest draw that delivers just as much.
    ///
    /// Zero when no two draws tie on throughput, which is the honest reading
    /// most of the time: the tiebreak can only pay out when there is a tie.
    ///
    /// Zero too when nothing delivers, for the same reason. The tiebreak declines
    /// to run at all when every draw is at 0.0 — see [`beats`] — so the winner is
    /// not the leanest of them and there is no saving to claim.
    pub fn parts_saved(&self) -> usize {
        let best = self.best_score();
        if best <= 0.0 {
            return 0;
        }
        let tied = self
            .scores
            .iter()
            .zip(&self.parts)
            .filter(|(&score, _)| score == best)
            .map(|(_, &parts)| parts);
        tied.max().unwrap_or(0) - self.best_parts()
    }
}

/// Draw `cfg.n` candidates per task, score each with the simulator, and keep the
/// best. Returns one [`BestOfN`] per input, in input order.
pub fn best_of_n<B: Backend>(
    model: &Denoiser<B>,
    partials: &[Grid],
    observed: &[Vec<bool>],
    cfg: &BestOfNConfig,
    device: &B::Device,
) -> Vec<BestOfN> {
    assert_eq!(partials.len(), observed.len());
    let tasks = partials.len();
    if tasks == 0 {
        return Vec::new();
    }
    let draws = cfg.n.max(1);

    let mut best: Vec<Option<ReconstructionDiagnostics>> = (0..tasks).map(|_| None).collect();
    let mut best_rank = vec![
        Ranked {
            buildable: false,
            score: f64::NEG_INFINITY,
            parts: usize::MAX,
        };
        tasks
    ];
    let mut scores: Vec<Vec<f64>> = (0..tasks).map(|_| Vec::with_capacity(draws)).collect();
    let mut part_counts: Vec<Vec<usize>> = (0..tasks).map(|_| Vec::with_capacity(draws)).collect();
    let mut buildable: Vec<Vec<bool>> = (0..tasks).map(|_| Vec::with_capacity(draws)).collect();
    let mut seen: Vec<HashSet<u64>> = (0..tasks).map(|_| HashSet::new()).collect();

    // One batched pass per draw, rather than a single batch of `n * tasks`: the
    // same total work, but peak memory stays at one validation batch.
    for i in 0..draws {
        let round = SampleConfig {
            seed: cfg.sample.seed.wrapping_add(i as u64),
            ..cfg.sample.clone()
        };
        let candidates = reconstruct_with_diagnostics(model, partials, observed, &round, device);
        for (task, candidate) in candidates.into_iter().enumerate() {
            let rank = Ranked {
                buildable: candidate.grid.is_consistent(),
                score: usable_score(&candidate.grid),
                parts: parts(&candidate.grid),
            };
            seen[task].insert(digest(&candidate.grid));
            scores[task].push(rank.score);
            part_counts[task].push(rank.parts);
            buildable[task].push(rank.buildable);

            if beats(rank, best_rank[task], cfg.prefer_compact) {
                best_rank[task] = rank;
                best[task] = Some(candidate);
            }
        }
    }

    best.into_iter()
        .enumerate()
        .map(|(task, best)| BestOfN {
            best: best.expect("every task gets at least one draw"),
            scores: std::mem::take(&mut scores[task]),
            parts: std::mem::take(&mut part_counts[task]),
            buildable: std::mem::take(&mut buildable[task]),
            distinct: seen[task].len(),
        })
        .collect()
}

/// A cheap identity for a candidate. Only the cells matter: draws for the same
/// task always share dimensions and obstacles.
fn digest(grid: &Grid) -> u64 {
    let mut hasher = DefaultHasher::new();
    for cell in &grid.cells {
        (
            cell.entity as u8,
            cell.direction as u8,
            cell.item as u8,
            cell.misc as u8,
        )
            .hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::factory_gen::{generate, Canvas, LessonKind};
    use crate::model::DenoiserConfig;
    use crate::sample::reconstruct;
    use crate::world::{Cell, Direction, Entity, Item};
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    type B = CpuBackend;

    /// A handful of blanked factories plus a small untrained denoiser. Untrained
    /// is fine and in fact preferable here: these tests are about the selection
    /// machinery, and a random model gives the widest spread of candidates.
    fn fixture(n: usize) -> (Denoiser<B>, Vec<Grid>, Vec<Vec<bool>>) {
        let device = Default::default();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (mut partials, mut observed) = (Vec::new(), Vec::new());
        for seed in 0..n as u64 {
            let sample = generate(LessonKind::MoveOneItem, Canvas::square(7), seed + 1).unwrap();
            let (partial, obs) = sample.blank(None, &mut rng);
            partials.push(partial);
            observed.push(obs);
        }
        let model = DenoiserConfig::new()
            .with_hidden(8)
            .with_blocks(1)
            .init::<B>(&device);
        (model, partials, observed)
    }

    /// The first draw is itself a candidate, so the winner can never be worse
    /// than what a single sample would have returned. This is the whole promise
    /// of Best-of-N and it must hold for every task, not on average.
    #[test]
    fn the_winner_is_never_worse_than_the_first_draw() {
        let (model, partials, observed) = fixture(4);
        let picks = best_of_n(
            &model,
            &partials,
            &observed,
            &BestOfNConfig {
                n: 6,
                sample: SampleConfig {
                    steps: 4,
                    temperature: 1.0,
                    seed: 5,
                },
                ..Default::default()
            },
            &Default::default(),
        );

        assert_eq!(picks.len(), 4);
        for pick in &picks {
            assert_eq!(pick.scores.len(), 6);
            assert!(
                pick.gain() >= 0.0,
                "best {} < first {}",
                pick.best_score(),
                pick.first_score()
            );
            assert_eq!(pick.best_score(), usable_score(&pick.best.grid));
            assert_eq!(pick.buildable.len(), 6);
        }
    }

    /// The returned grid must be the one that scored best — not merely *a* draw
    /// with the best score attached to it.
    #[test]
    fn the_returned_grid_is_the_one_that_scored_best() {
        let (model, partials, observed) = fixture(3);
        let cfg = BestOfNConfig {
            n: 5,
            sample: SampleConfig {
                steps: 4,
                temperature: 1.2,
                seed: 21,
            },
            ..Default::default()
        };
        let picks = best_of_n(&model, &partials, &observed, &cfg, &Default::default());

        // Redraw the same candidates by hand and confirm the winner is the max.
        for (task, pick) in picks.iter().enumerate() {
            let mut replayed = Vec::new();
            for i in 0..cfg.n {
                let round = SampleConfig {
                    seed: cfg.sample.seed + i as u64,
                    ..cfg.sample.clone()
                };
                let grids = reconstruct(&model, &partials, &observed, &round, &Default::default());
                replayed.push(usable_score(&grids[task]));
            }
            assert_eq!(replayed, pick.scores, "draws are not reproducible");
            assert_eq!(
                pick.best_score(),
                replayed.iter().copied().fold(f64::MIN, f64::max)
            );
        }
    }

    /// Best-of-N is inpainting too: the given cells are the task statement and no
    /// candidate — winner included — is allowed to rewrite them.
    #[test]
    fn the_winner_still_preserves_the_observed_cells() {
        let (model, partials, observed) = fixture(2);
        let picks = best_of_n(
            &model,
            &partials,
            &observed,
            &BestOfNConfig {
                n: 4,
                sample: SampleConfig {
                    steps: 4,
                    temperature: 1.0,
                    seed: 3,
                },
                ..Default::default()
            },
            &Default::default(),
        );

        for (pick, (partial, obs)) in picks.iter().zip(partials.iter().zip(&observed)) {
            for (cell, &is_observed) in obs.iter().enumerate() {
                if is_observed {
                    let (x, y) = (cell % partial.width, cell / partial.width);
                    assert_eq!(pick.best.grid.get(x, y), partial.get(x, y));
                }
            }
        }
    }

    /// With `n = 1` and no temperature there is nothing to choose between, so
    /// Best-of-N must return exactly what the plain sampler does. Guards against
    /// the selection layer quietly changing single-shot behaviour.
    #[test]
    fn a_single_greedy_draw_is_just_plain_sampling() {
        let (model, partials, observed) = fixture(2);
        let sample = SampleConfig {
            steps: 5,
            temperature: 0.0,
            seed: 9,
        };
        let picks = best_of_n(
            &model,
            &partials,
            &observed,
            &BestOfNConfig {
                n: 1,
                sample: sample.clone(),
                ..Default::default()
            },
            &Default::default(),
        );
        let plain = reconstruct(&model, &partials, &observed, &sample, &Default::default());

        for (pick, grid) in picks.iter().zip(&plain) {
            assert_eq!(&pick.best.grid, grid);
            assert_eq!(pick.distinct, 1);
            assert_eq!(pick.gain(), 0.0);
        }
    }

    /// Temperature is what makes the draws differ, and differing draws are what
    /// Best-of-N spends its compute on. If greedy decoding silently produced
    /// variety — or temperature silently produced none — the knob would be lying.
    #[test]
    fn temperature_is_what_produces_candidates_to_choose_between() {
        let (model, partials, observed) = fixture(4);
        let draw = |temperature| {
            best_of_n(
                &model,
                &partials,
                &observed,
                &BestOfNConfig {
                    n: 4,
                    sample: SampleConfig {
                        steps: 4,
                        temperature,
                        seed: 4,
                    },
                    ..Default::default()
                },
                &Default::default(),
            )
        };

        // Greedy: four passes, one answer, four times over.
        assert!(draw(0.0).iter().all(|p| p.distinct == 1));
        // Hot: the model is untrained, so every draw should land somewhere new.
        assert!(draw(1.5).iter().all(|p| p.distinct > 1));
    }

    /// Two factories that deliver the same items/second, one built from fewer
    /// parts. This is the whole of what "rewarded for compactness" can safely
    /// mean here, so it is worth stating on layouts we control rather than on
    /// whatever an untrained model happens to draw.
    ///
    /// A belt run of any length carries a full belt, so the short way and the
    /// long way around score identically and only the part count separates them.
    fn anchor(entity: Entity, item: Item) -> Cell {
        Cell {
            entity,
            item,
            ..Default::default()
        }
    }

    fn detour(long: bool) -> Grid {
        //     S > > > K        or        S > > > K
        //     . . . . .                  . ^ . . ^
        //     . . . . .                  . < < < <
        let mut g = Grid::new(5, 3);
        g.set(0, 0, anchor(Entity::Source, Item::IronPlate));
        g.set(4, 0, anchor(Entity::Sink, Item::IronPlate));
        if long {
            g.set(1, 0, Cell::belt(Direction::South));
            for x in 1..4 {
                g.set(x, 2, Cell::belt(Direction::East));
            }
            g.set(1, 1, Cell::belt(Direction::South));
            g.set(4, 2, Cell::belt(Direction::North));
            g.set(4, 1, Cell::belt(Direction::North));
        } else {
            for x in 1..4 {
                g.set(x, 0, Cell::belt(Direction::East));
            }
        }
        g
    }

    #[test]
    fn the_long_way_round_and_the_short_way_deliver_the_same() {
        assert_eq!(
            throughput::score(&detour(false)),
            throughput::score(&detour(true))
        );
        assert!(throughput::score(&detour(false)) > 0.0);
        assert!(parts(&detour(true)) > parts(&detour(false)));
    }

    /// The trap in rewarding compactness: the most compact factory of all is the
    /// empty one. It must never win, however many parts it saves — so the
    /// tiebreak only ever runs among candidates that already deliver.
    #[test]
    fn an_empty_factory_never_wins_on_being_compact() {
        let empty = Grid::new(5, 3);
        assert_eq!(parts(&empty), 0);
        assert_eq!(throughput::score(&empty), 0.0);
        // Nothing is more compact, and nothing is worth less.
        assert!(parts(&empty) < parts(&detour(false)));
        assert!(throughput::score(&empty) < throughput::score(&detour(false)));

        // Scored in draw order, the empty grid arrives first and still loses.
        assert_eq!(
            pick_index(&[ranked(&empty), ranked(&detour(false))], true),
            1
        );
    }

    /// Which draw wins, run through the same [`beats`] the draw loop uses, and
    /// seeded the same way, so this cannot drift from what `best_of_n` does.
    fn pick_index(draws: &[Ranked], prefer_compact: bool) -> usize {
        let mut winner = 0;
        let mut best = Ranked {
            buildable: false,
            score: f64::NEG_INFINITY,
            parts: usize::MAX,
        };
        for (i, &cand) in draws.iter().enumerate() {
            if beats(cand, best, prefer_compact) {
                winner = i;
                best = cand;
            }
        }
        winner
    }

    /// A grid reduced to what ranking sees, exactly as the draw loop reduces it.
    fn ranked(grid: &Grid) -> Ranked {
        Ranked {
            buildable: grid.is_consistent(),
            score: usable_score(grid),
            parts: parts(grid),
        }
    }

    /// A belt facing nowhere: the cell the user's log kept refusing to export.
    fn belt_facing_nowhere() -> Cell {
        Cell {
            entity: Entity::TransportBelt,
            direction: Direction::None,
            ..Default::default()
        }
    }

    #[test]
    fn a_tie_on_throughput_goes_to_the_factory_with_fewer_parts() {
        let draws = [ranked(&detour(true)), ranked(&detour(false))];

        // The roomy detour is drawn first and is beaten by the lean one behind it.
        assert_eq!(pick_index(&draws, true), 1);
        // Switched off, the first of the tied draws stands: this is the choice
        // the flag makes, not something the scores were going to decide anyway.
        assert_eq!(pick_index(&draws, false), 0);
    }

    /// Compactness is a tiebreak, never a trade: a leaner factory that delivers
    /// less must lose, or the flag would be quietly costing throughput.
    #[test]
    fn compactness_never_outranks_throughput() {
        let draws = [
            Ranked {
                buildable: true,
                score: 15.0,
                parts: 12,
            },
            Ranked {
                buildable: true,
                score: 0.86,
                parts: 3,
            },
        ];
        assert_eq!(pick_index(&draws, true), 0);
    }

    /// The user's import error, in grid form. The simulator is happy to score the
    /// factory around a belt facing nowhere — the belt run beside it really would
    /// deliver — but Factorio rejects the blueprint, so the whole thing delivers
    /// nothing. This is the gap [`usable_score`] closes.
    #[test]
    fn a_factory_that_cannot_be_built_delivers_nothing() {
        let mut broken = detour(false);
        broken.set(0, 2, belt_facing_nowhere());
        assert!(
            !broken.is_consistent(),
            "a belt facing nowhere is not legal"
        );

        // The raw figure is not wrong about the flow; it is about a factory that
        // cannot exist. Unchanged, because the stray belt is off the path.
        assert_eq!(
            throughput::score(&broken),
            throughput::score(&detour(false))
        );
        assert!(throughput::score(&broken) > 0.0);
        assert_eq!(usable_score(&broken), 0.0);
    }

    /// The path that handed the user an unimportable layout even when a fine one
    /// was drawn. `best` starts at negative infinity, so draw 0 took it; with
    /// every draw at 0.0 nothing behind it could pass `score > best`, and the
    /// compactness tiebreak declines to run at zero. Draw 0 stood unconditionally
    /// and the slate behind it went unread.
    #[test]
    fn an_unbuildable_draw_never_wins_however_it_is_drawn() {
        let empty = Grid::new(5, 3);
        let mut broken = Grid::new(5, 3);
        broken.set(0, 2, belt_facing_nowhere());
        assert_eq!(usable_score(&broken), 0.0);
        assert_eq!(usable_score(&empty), 0.0);

        // Both deliver nothing and the unbuildable one is drawn first. The empty
        // grid is a real factory that happens to be useless, which is strictly
        // more than the other one is, so it takes it.
        assert_eq!(pick_index(&[ranked(&broken), ranked(&empty)], true), 1);

        // Against a factory that works it loses from either side of the draw.
        let working = detour(false);
        assert_eq!(pick_index(&[ranked(&broken), ranked(&working)], true), 1);
        assert_eq!(pick_index(&[ranked(&working), ranked(&broken)], true), 0);
    }

    /// Buildable outranks throughput, and not merely as a tiebreak: this draw
    /// simulates better *and* is leaner, and still loses.
    ///
    /// [`usable_score`] would already have zeroed that 15.0, so this state is one
    /// the draw loop cannot reach — which is the point. The two guards are
    /// independent, and [`beats`] must not start depending on the other one.
    #[test]
    fn a_draw_that_cannot_be_built_loses_however_well_it_simulates() {
        let draws = [
            Ranked {
                buildable: false,
                score: 15.0,
                parts: 3,
            },
            Ranked {
                buildable: true,
                score: 0.86,
                parts: 12,
            },
        ];
        assert_eq!(pick_index(&draws, true), 1);
    }

    #[test]
    fn the_compactness_tiebreak_reports_the_parts_it_saved() {
        let (model, partials, observed) = fixture(3);
        let picks = best_of_n(
            &model,
            &partials,
            &observed,
            &BestOfNConfig {
                n: 6,
                sample: SampleConfig {
                    steps: 4,
                    temperature: 1.0,
                    seed: 5,
                },
                prefer_compact: true,
            },
            &Default::default(),
        );

        for pick in &picks {
            assert_eq!(pick.parts.len(), pick.scores.len());
            assert_eq!(pick.best_parts(), parts(&pick.best.grid));
            // The winner is the leanest of everything that tied with it, so it
            // can never be beaten on parts by another draw at the same score.
            let best = pick.best_score();
            for (&score, &count) in pick.scores.iter().zip(&pick.parts) {
                if score == best && best > 0.0 {
                    assert!(pick.best_parts() <= count);
                }
            }
        }
    }
}
