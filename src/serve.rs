//! Paint a task, watch the model design it — over HTTP, at runtime.
//!
//! `experiments/gallery.rs` renders a fixed page of *generated* lessons: the
//! tasks come from [`factory_gen`](crate::factory_gen), so the only factories
//! you can look at are the ones the curriculum already knows how to pose. That
//! answers "did the model learn the lessons" and nothing else. The question this
//! module exists for is the other one: **give the model a task nobody generated
//! and see what it builds.**
//!
//! So the unit here is a [`TaskSpec`] — a bare rectangle with the ports pinned
//! to its edges ("plates enter here, gears must leave there") and everything
//! else masked. That is deliberately the same shape as
//! [`Sample::blank_to_scaffold`](crate::factory_gen::Sample::blank_to_scaffold),
//! and the same shape the planned mod poses in-game: select an area, bring the
//! inputs and outputs to its edge, ask for the middle.
//!
//! Two halves, split so the interesting one is testable without a socket:
//!
//! * [`design`] — spec in, [`DesignResponse`] out. Pure, no I/O.
//! * [`serve`] — a minimal HTTP/1.1 loop that parses a request, calls [`design`]
//!   and writes JSON back.
//!
//! The server binds loopback and speaks just enough HTTP for a `fetch` from the
//! page it serves. It is a development viewer, not a deployment: single
//! threaded, no TLS, no auth — anything that can reach the port can spend your
//! CPU on a denoising run.

use crate::blueprint::{blueprint_string, grid_to_blueprint};
use crate::model::Denoiser;
use crate::sample::SampleConfig;
use crate::viewer::grid_to_svg;
use crate::world::{Cell, Direction, Entity, Grid, Item, Misc};
use crate::{best_of_n, sim, textual, throughput};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// Largest grid a request may ask for. A denoising run is `steps` forward passes
/// over `width × height` cells and the server is single threaded, so this is
/// what keeps one careless request from wedging the viewer for a minute.
pub const MAX_SIDE: usize = 32;
/// Smallest grid worth posing: a 3×3 assembler plus a tile of room either side.
pub const MIN_SIDE: usize = 5;
/// Ceiling on best-of-N draws per request, for the same reason as [`MAX_SIDE`].
pub const MAX_CANDIDATES: usize = 32;
/// Ceiling on denoising rounds per request.
pub const MAX_STEPS: usize = 64;
/// Largest request body accepted, in bytes. A spec is a few hundred cells of
/// JSON; anything larger is a mistake or an attempt to exhaust memory.
pub const MAX_BODY: usize = 1 << 20;

/// One cell the user pinned. Everything not pinned is the model's to fill.
///
/// `direction`/`item`/`misc` default to `None`, so pinning a plain source is
/// `{"x": 0, "y": 4, "entity": "Source", "item": "IronPlate"}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Pin {
    pub x: usize,
    pub y: usize,
    pub entity: Entity,
    #[serde(default = "no_direction")]
    pub direction: Direction,
    #[serde(default = "no_item")]
    pub item: Item,
    #[serde(default = "no_misc")]
    pub misc: Misc,
}

fn no_direction() -> Direction {
    Direction::None
}
fn no_item() -> Item {
    Item::None
}
fn no_misc() -> Misc {
    Misc::None
}

/// A task posed at runtime: a rectangle, the cells the user pinned, and the
/// terrain they must build around.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskSpec {
    pub width: usize,
    pub height: usize,
    /// Cells held fixed. The model conditions on these and may not overwrite
    /// them — usually the ports, but anything may be pinned, which is what makes
    /// this an editor rather than a lesson picker.
    #[serde(default)]
    pub pins: Vec<Pin>,
    /// Terrain. Obstacles live on their own conditioning plane and are never
    /// generated, matching how the lessons pose them.
    #[serde(default)]
    pub obstacles: Vec<[usize; 2]>,
    /// Best-of-N draws. `1` is a single greedy shot.
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Above zero the draws differ; at zero every draw is the same argmax, so
    /// best-of-N degenerates to one shot however many candidates are asked for.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub seed: u64,
}

fn default_candidates() -> usize {
    8
}
fn default_steps() -> usize {
    12
}
fn default_temperature() -> f64 {
    0.9
}

/// What one sink got, flattened for the page.
#[derive(Debug, Clone, Serialize)]
pub struct Delivery {
    pub x: usize,
    pub y: usize,
    pub item: Item,
    pub achieved: f64,
}

/// The model's answer, plus every number needed to judge it without trusting it.
#[derive(Debug, Clone, Serialize)]
pub struct DesignResponse {
    /// The finished factory.
    pub svg: String,
    /// One SVG per denoising round, cells appearing as they are committed.
    ///
    /// Faithful rather than decorative: the sampler freezes a cell when it
    /// commits it, so a frame really is what the model had decided by that
    /// round. See [`frames`].
    pub frames: Vec<String>,
    /// The ASCII render, for copying into an issue.
    pub ascii: String,
    /// Items/second the simulator measures — the honest verdict on the picture.
    pub score: f64,
    /// Whether anything reaches a sink at all. A factory can be `true` here and
    /// still score `0.0`; the two answer different questions.
    pub functional: bool,
    pub deliveries: Vec<Delivery>,
    /// Score of every draw, in draw order, so the page can show what best-of-N
    /// rejected and whether the spread is real.
    pub candidate_scores: Vec<f64>,
    /// Distinct grids among the draws. `1` means the model had one idea.
    pub distinct: usize,
    /// Importable into Factorio, or the reason it is not.
    pub blueprint: Option<String>,
    pub blueprint_error: Option<String>,
}

/// Why a spec was refused. Bad input is the normal case for a hand-painted task,
/// so it gets a type rather than a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    Size { side: usize },
    OutOfBounds { x: usize, y: usize },
    Overlap { x: usize, y: usize },
    NoSource,
    NoSink,
    Unbuildable,
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Size { side } => write!(
                f,
                "grid sides must be {MIN_SIDE}..={MAX_SIDE} tiles, got {side}"
            ),
            Self::OutOfBounds { x, y } => write!(f, "({x}, {y}) is outside the grid"),
            Self::Overlap { x, y } => write!(f, "({x}, {y}) is pinned twice"),
            Self::NoSource => write!(f, "pin at least one Source: the model needs an input"),
            Self::NoSink => write!(f, "pin at least one Sink: the model needs somewhere to deliver"),
            Self::Unbuildable => write!(
                f,
                "the pinned cells overlap each other's footprints, so no legal factory contains them"
            ),
        }
    }
}

impl std::error::Error for SpecError {}

impl TaskSpec {
    /// Turn a painted spec into the `(partial, observed)` pair the sampler
    /// takes, or say why it cannot be one.
    ///
    /// The mask mirrors `blank_to_scaffold`: pinned cells are observed, and
    /// *everything else is masked* — including the tiles under obstacles, which
    /// the model must learn to leave alone rather than be told to.
    pub fn to_task(&self) -> Result<(Grid, Vec<bool>), SpecError> {
        for side in [self.width, self.height] {
            if !(MIN_SIDE..=MAX_SIDE).contains(&side) {
                return Err(SpecError::Size { side });
            }
        }
        let mut grid = Grid::new(self.width, self.height);
        for &[x, y] in &self.obstacles {
            if x >= self.width || y >= self.height {
                return Err(SpecError::OutOfBounds { x, y });
            }
            grid.set_obstacle(x, y, true);
        }

        let mut observed = vec![false; grid.len()];
        for pin in &self.pins {
            if pin.x >= self.width || pin.y >= self.height {
                return Err(SpecError::OutOfBounds { x: pin.x, y: pin.y });
            }
            let i = grid.idx(pin.x, pin.y);
            if observed[i] {
                return Err(SpecError::Overlap { x: pin.x, y: pin.y });
            }
            observed[i] = true;
            grid.set(
                pin.x,
                pin.y,
                Cell {
                    entity: pin.entity,
                    direction: pin.direction,
                    item: pin.item,
                    misc: pin.misc,
                },
            );
        }

        let has = |e: Entity| grid.cells.iter().any(|c| c.entity == e);
        if !has(Entity::Source) {
            return Err(SpecError::NoSource);
        }
        if !has(Entity::Sink) {
            return Err(SpecError::NoSink);
        }
        // Two 3×3 machines pinned two tiles apart can never both stand, and the
        // model would be asked for a factory that does not exist. Catch it here
        // rather than letting it surface as a mystified zero score.
        if !grid.footprints_are_legal() {
            return Err(SpecError::Unbuildable);
        }
        Ok((grid, observed))
    }
}

/// Rebuild the grid as it stood after round `round`.
///
/// Cells the sampler had not yet committed read back as empty. This is honest
/// only because commitment is final: [`crate::sample`] freezes a cell once it
/// unmasks it, so no frame ever shows a cell that a later round overwrites.
/// Obstacles are terrain and are present from the first frame.
pub fn frame_at(final_grid: &Grid, reveal_step: &[usize], round: usize) -> Grid {
    let mut g = Grid::new(final_grid.width, final_grid.height);
    for y in 0..final_grid.height {
        for x in 0..final_grid.width {
            g.set_obstacle(x, y, final_grid.is_obstacle(x, y));
        }
    }
    for (i, &step) in reveal_step.iter().enumerate() {
        if step <= round {
            g.cells[i] = final_grid.cells[i];
        }
    }
    g
}

/// One SVG per round, from the conditioning-only frame to the finished factory.
fn frames(grid: &Grid, reveal_step: &[usize]) -> Vec<String> {
    let last = reveal_step.iter().copied().max().unwrap_or(0);
    (0..=last)
        .map(|r| grid_to_svg(&frame_at(grid, reveal_step, r)))
        .collect()
}

/// Run the model on a painted task.
///
/// The whole point of the module, and deliberately free of I/O so a test can
/// pose a task without opening a socket.
pub fn design<B: Backend>(
    model: &Denoiser<B>,
    spec: &TaskSpec,
    device: &B::Device,
) -> Result<DesignResponse, SpecError> {
    let (partial, observed) = spec.to_task()?;

    let cfg = best_of_n::BestOfNConfig {
        n: spec.candidates.clamp(1, MAX_CANDIDATES),
        sample: SampleConfig {
            steps: spec.steps.clamp(1, MAX_STEPS),
            temperature: spec.temperature.max(0.0),
            seed: spec.seed,
        },
    };
    let mut runs = best_of_n::best_of_n(model, &[partial], &[observed], &cfg, device);
    let run = runs.pop().expect("one task in, one result out");
    let grid = run.best.grid;

    let report = throughput::throughput(&grid);
    let (blueprint, blueprint_error) = match grid_to_blueprint(&grid, "designed at runtime")
        .and_then(|bp| blueprint_string(&bp))
    {
        Ok(s) => (Some(s), None),
        // A model early in training draws factories Factorio would refuse to
        // import. That is a fact about the factory worth showing, not an error
        // worth failing the request over.
        Err(e) => (None, Some(e.to_string())),
    };

    Ok(DesignResponse {
        svg: grid_to_svg(&grid),
        frames: frames(&grid, &run.best.reveal_step),
        ascii: textual::render(&grid),
        score: report.score,
        functional: sim::item_reaches_sink(&grid),
        deliveries: report
            .deliveries
            .iter()
            .map(|d| Delivery {
                x: d.at.0,
                y: d.at.1,
                item: d.item,
                achieved: d.achieved,
            })
            .collect(),
        candidate_scores: run.scores,
        distinct: run.distinct,
        blueprint,
        blueprint_error,
    })
}

/// The editor page.
const INDEX_HTML: &str = include_str!("serve/index.html");

/// Serve the viewer on `addr` until killed.
///
/// Single threaded on purpose: one denoising run at a time keeps the box
/// responsive and the code short. Requests queue.
pub fn serve<B: Backend>(
    model: &Denoiser<B>,
    addr: &str,
    device: &B::Device,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let local = listener.local_addr()?;
    println!("viewer on http://{local}  (ctrl-c to stop)");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                // One bad request must not take the server down with it.
                if let Err(e) = handle(model, s, device) {
                    eprintln!("request failed: {e}");
                }
            }
            Err(e) => eprintln!("connection failed: {e}"),
        }
    }
    Ok(())
}

fn handle<B: Backend>(
    model: &Denoiser<B>,
    mut stream: TcpStream,
    device: &B::Device,
) -> std::io::Result<()> {
    let Some(request) = read_request(&mut stream)? else {
        return Ok(());
    };
    let (status, content_type, body) = route(model, &request, device);
    write_response(&mut stream, status, content_type, &body)
}

/// As much of a request as this server cares about.
struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// Parse a request off the wire, or `None` if the peer sent nothing usable.
fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        return Ok(None);
    };
    let (method, path) = (method.to_string(), path.to_string());

    let mut length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some(v) = header
            .split_once(':')
            .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.trim())
        {
            length = v.parse().unwrap_or(0);
        }
    }

    // Read no more than advertised, and never more than the cap: `take` is what
    // stops a lying Content-Length from allocating the machine's memory.
    let mut body = Vec::new();
    if length > 0 {
        reader
            .by_ref()
            .take(length.min(MAX_BODY) as u64)
            .read_to_end(&mut body)?;
    }
    Ok(Some(Request { method, path, body }))
}

fn route<B: Backend>(
    model: &Denoiser<B>,
    request: &Request,
    device: &B::Device,
) -> (&'static str, &'static str, Vec<u8>) {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => (
            "200 OK",
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes().to_vec(),
        ),
        ("POST", "/api/design") => {
            if request.body.len() >= MAX_BODY {
                return json_error("413 Payload Too Large", "task spec too large");
            }
            let spec: TaskSpec = match serde_json::from_slice(&request.body) {
                Ok(s) => s,
                Err(e) => return json_error("400 Bad Request", &e.to_string()),
            };
            match design(model, &spec, device) {
                Ok(r) => match serde_json::to_vec(&r) {
                    Ok(b) => ("200 OK", "application/json", b),
                    Err(e) => json_error("500 Internal Server Error", &e.to_string()),
                },
                Err(e) => json_error("400 Bad Request", &e.to_string()),
            }
        }
        _ => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found".to_vec(),
        ),
    }
}

fn json_error(status: &'static str, message: &str) -> (&'static str, &'static str, Vec<u8>) {
    let body = serde_json::json!({ "error": message });
    (
        status,
        "application/json",
        serde_json::to_vec(&body).unwrap_or_else(|_| b"{\"error\":\"unprintable\"}".to_vec()),
    )
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Direction;

    fn pin(x: usize, y: usize, entity: Entity, item: Item) -> Pin {
        Pin {
            x,
            y,
            entity,
            direction: Direction::None,
            item,
            misc: Misc::None,
        }
    }

    /// The smallest useful spec: plates in on one edge, gears out on the other.
    fn plates_to_gears() -> TaskSpec {
        TaskSpec {
            width: 11,
            height: 7,
            pins: vec![
                pin(0, 3, Entity::Source, Item::IronPlate),
                pin(10, 3, Entity::Sink, Item::IronGear),
            ],
            obstacles: vec![],
            candidates: 4,
            steps: 8,
            temperature: 0.9,
            seed: 0,
        }
    }

    #[test]
    fn a_spec_masks_everything_it_did_not_pin() {
        let (grid, observed) = plates_to_gears().to_task().expect("valid");
        assert_eq!(observed.iter().filter(|o| **o).count(), 2);
        assert!(observed[grid.idx(0, 3)]);
        assert!(observed[grid.idx(10, 3)]);
        assert_eq!(grid.get(0, 3).entity, Entity::Source);
        assert_eq!(grid.get(10, 3).item, Item::IronGear);
    }

    /// Obstacles condition but are not given: the model has to work out that it
    /// may not build there, exactly as `blank_to_scaffold` poses it.
    #[test]
    fn an_obstacle_is_terrain_rather_than_a_given_cell() {
        let mut spec = plates_to_gears();
        spec.obstacles = vec![[5, 3]];
        let (grid, observed) = spec.to_task().expect("valid");
        assert!(grid.is_obstacle(5, 3));
        assert!(!observed[grid.idx(5, 3)]);
    }

    #[test]
    fn a_task_without_ports_is_refused_rather_than_guessed_at() {
        let mut spec = plates_to_gears();
        spec.pins.retain(|p| p.entity != Entity::Sink);
        assert_eq!(spec.to_task(), Err(SpecError::NoSink));

        let mut spec = plates_to_gears();
        spec.pins.clear();
        assert_eq!(spec.to_task(), Err(SpecError::NoSource));
    }

    #[test]
    fn a_grid_too_big_to_denoise_promptly_is_refused() {
        let mut spec = plates_to_gears();
        spec.width = MAX_SIDE + 1;
        assert_eq!(spec.to_task(), Err(SpecError::Size { side: MAX_SIDE + 1 }));
        spec.width = 2;
        assert_eq!(spec.to_task(), Err(SpecError::Size { side: 2 }));
    }

    #[test]
    fn pinning_the_same_cell_twice_is_refused() {
        let mut spec = plates_to_gears();
        spec.pins.push(pin(0, 3, Entity::Sink, Item::IronGear));
        assert_eq!(spec.to_task(), Err(SpecError::Overlap { x: 0, y: 3 }));
    }

    /// Two machines pinned so close that their 3×3 bodies collide describe a
    /// factory that cannot exist. Better to say so than to hand the model an
    /// impossible task and let it look incompetent.
    #[test]
    fn pinned_footprints_that_collide_are_refused() {
        let mut spec = plates_to_gears();
        spec.pins.push(Pin {
            x: 3,
            y: 2,
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::IronGear,
            misc: Misc::None,
        });
        spec.pins.push(Pin {
            x: 4,
            y: 2,
            entity: Entity::Assembler,
            direction: Direction::East,
            item: Item::IronGear,
            misc: Misc::None,
        });
        assert_eq!(spec.to_task(), Err(SpecError::Unbuildable));
    }

    /// A pin lands in the grid whatever else is masked, and `x`/`y` are not
    /// swapped on the way in — the sort of thing a rectangle catches and a
    /// square never would.
    #[test]
    fn a_pin_lands_where_it_was_painted_on_a_non_square_grid() {
        let spec = TaskSpec {
            width: 13,
            height: 7,
            pins: vec![pin(11, 1, Entity::Sink, Item::IronGear)],
            ..plates_to_gears()
        };
        let (grid, _) = TaskSpec {
            pins: {
                let mut p = spec.pins.clone();
                p.push(pin(0, 5, Entity::Source, Item::IronPlate));
                p
            },
            ..spec
        }
        .to_task()
        .expect("valid");
        assert_eq!(grid.get(11, 1).entity, Entity::Sink);
        assert_eq!(grid.get(0, 5).entity, Entity::Source);
    }

    /// The animation's contract: frame `r` holds every cell committed by round
    /// `r` and nothing else, frame 0 is the conditioning the user painted, and
    /// the last frame is the finished factory.
    #[test]
    fn a_frame_holds_exactly_the_cells_committed_by_that_round() {
        let mut grid = Grid::new(5, 1);
        grid.set(
            0,
            0,
            Cell {
                entity: Entity::Source,
                item: Item::IronPlate,
                ..Default::default()
            },
        );
        grid.set(1, 0, Cell::belt(Direction::East));
        grid.set(2, 0, Cell::belt(Direction::East));
        grid.set_obstacle(3, 0, true);
        grid.set(
            4,
            0,
            Cell {
                entity: Entity::Sink,
                item: Item::IronPlate,
                ..Default::default()
            },
        );
        // Ports observed (0), then one belt per round.
        let reveal = vec![0, 1, 2, 0, 0];

        let f0 = frame_at(&grid, &reveal, 0);
        assert_eq!(f0.get(0, 0).entity, Entity::Source);
        assert_eq!(f0.get(4, 0).entity, Entity::Sink);
        assert!(
            f0.get(1, 0).is_empty(),
            "belt was not committed until round 1"
        );
        assert!(
            f0.is_obstacle(3, 0),
            "terrain is there from the first frame"
        );

        let f1 = frame_at(&grid, &reveal, 1);
        assert_eq!(f1.get(1, 0).entity, Entity::TransportBelt);
        assert!(f1.get(2, 0).is_empty());

        assert_eq!(frame_at(&grid, &reveal, 2).cells, grid.cells);
    }

    #[test]
    fn there_is_one_frame_per_round_plus_the_conditioning() {
        let mut grid = Grid::new(3, 1);
        grid.set(0, 0, Cell::belt(Direction::East));
        let svgs = frames(&grid, &[0, 3, 1]);
        assert_eq!(svgs.len(), 4, "rounds 0..=3");
        assert!(svgs[0].starts_with("<svg"));
    }

    /// Defaults exist so the page can post `{"entity": "Source"}` without
    /// spelling out every channel.
    #[test]
    fn a_pin_needs_only_a_position_and_an_entity() {
        let spec: TaskSpec = serde_json::from_str(
            r#"{"width": 11, "height": 7,
                "pins": [{"x": 0, "y": 3, "entity": "Source", "item": "IronPlate"},
                         {"x": 10, "y": 3, "entity": "Sink", "item": "IronGear"}]}"#,
        )
        .expect("parses");
        assert_eq!(spec.candidates, default_candidates());
        assert_eq!(spec.steps, default_steps());
        assert_eq!(spec.pins[0].direction, Direction::None);
        assert_eq!(spec.pins[0].misc, Misc::None);
        spec.to_task().expect("valid");
    }
}
