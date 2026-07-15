use diffusion_factorio::metrics::ReconReport;
use diffusion_factorio::observability::{
    write_metrics_jsonl, write_sample_report, write_training_report, RunMetadata, SampleReportEntry,
};
use diffusion_factorio::train::TrainLog;
use diffusion_factorio::world::{Cell, Direction, Grid};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "diffusion-factorio-{name}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

fn example_logs() -> Vec<TrainLog> {
    vec![
        TrainLog {
            step: 1,
            lr: 3e-6,
            loss: 6.3,
            train_acc: [0.03, 0.05, 0.66, 0.76],
            placement_acc: 0.06,
            t_mean: 0.5,
            nll: 1.2,
            channel_nll: [0.4, 0.3, 0.2, 0.3],
            elapsed_seconds: 0.25,
            samples_seen: 32,
            samples_per_second: 128.0,
            val: None,
            val_by_lesson: Default::default(),
            val_scratch: None,
            val_scratch_by_lesson: Default::default(),
        },
        TrainLog {
            step: 200,
            lr: 3e-4,
            loss: 0.76,
            train_acc: [0.91, 0.93, 1.0, 1.0],
            placement_acc: 0.79,
            t_mean: 0.45,
            nll: 0.31,
            channel_nll: [0.1, 0.08, 0.06, 0.07],
            elapsed_seconds: 25.0,
            samples_seen: 6400,
            samples_per_second: 256.0,
            val: Some(ReconReport {
                n_factories: 64,
                masked_cells: 100,
                channel_correct: [90, 91, 100, 99],
                exact: 20,
                consistent: 60,
                functional: 21,
                original_functional: 64,
            }),
            val_by_lesson: Default::default(),
            // From scratch the whole factory is masked, so far more cells are
            // scored and far fewer runs deliver the item.
            val_scratch: Some(ReconReport {
                n_factories: 64,
                masked_cells: 7000,
                channel_correct: [6300, 6100, 6900, 6800],
                exact: 1,
                consistent: 30,
                functional: 5,
                original_functional: 64,
            }),
            val_scratch_by_lesson: BTreeMap::from([(
                "move_one_item".to_owned(),
                ReconReport {
                    n_factories: 16,
                    masked_cells: 1800,
                    channel_correct: [1700, 1650, 1790, 1780],
                    exact: 1,
                    consistent: 9,
                    functional: 4,
                    original_functional: 16,
                },
            )]),
        },
    ]
}

#[test]
fn metrics_jsonl_is_structured_and_keeps_validation() {
    let path = temp_file("metrics.jsonl");
    write_metrics_jsonl(&path, &example_logs()).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["step"], 1);
    assert_eq!(rows[1]["samples_seen"], 6400);
    assert_eq!(rows[1]["val"]["functional_rate"], 21.0 / 64.0);
    assert_eq!(rows[1]["val"]["entity_acc"], 0.9);
    fs::remove_file(path).ok();
}

/// The from-scratch pass is the one that answers "can it design a factory?", so
/// it has to reach the durable metrics rather than only the progress line.
#[test]
fn metrics_jsonl_keeps_the_from_scratch_pass_separate_from_inpainting() {
    let path = temp_file("metrics-scratch.jsonl");
    write_metrics_jsonl(&path, &example_logs()).unwrap();
    let rows: Vec<serde_json::Value> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    assert!(rows[0]["val_scratch"].is_null(), "no validation on step 1");
    assert_eq!(rows[1]["val_scratch"]["functional_rate"], 5.0 / 64.0);
    assert_eq!(
        rows[1]["val_scratch_by_lesson"]["move_one_item"]["functional_rate"],
        4.0 / 16.0
    );
    // Same factories, same n: only the conditioning differs, and building the
    // whole factory is strictly harder than filling its gaps.
    assert_eq!(rows[1]["val_scratch"]["n"], rows[1]["val"]["n"]);
    assert!(
        rows[1]["val_scratch"]["masked_cells"].as_u64() > rows[1]["val"]["masked_cells"].as_u64()
    );
    fs::remove_file(path).ok();
}

#[test]
fn html_report_embeds_parameters_and_metric_charts() {
    let path = temp_file("training-report.html");
    let metadata = RunMetadata {
        backend: "ndarray (CPU)".into(),
        grid_size: 11,
        steps: 200,
        batch_size: 32,
        val_batch: 64,
        sample_steps: 12,
        seed: 7,
        peak_lr: 3e-4,
        warmup_steps: 100,
        grad_clip: 1.0,
        hidden: 64,
        blocks: 6,
        embed_dim: 16,
        time_dim: 64,
        elbo_weight: false,
        t_min: 0.02,
        structure_weight: 8.0,
    };
    write_training_report(&path, &metadata, &example_logs()).unwrap();
    let html = fs::read_to_string(&path).unwrap();

    assert!(html.contains("Training report"));
    assert!(html.contains("structure_weight"));
    assert!(html.contains("Functional / exact / consistent"));
    assert!(html.contains("Built from scratch"));
    assert!(html.contains("Placement recall"));
    assert!(html.contains("per-channel NLL"));
    assert!(html.contains("application/json"));
    assert!(!html.contains("https://"), "report must work offline");
    fs::remove_file(path).ok();
}

#[test]
fn spatial_report_contains_confidence_entropy_and_error_heatmaps() {
    let path = temp_file("sample-report.html");
    let mut target = Grid::new(2, 1);
    target.set(0, 0, Cell::belt(Direction::East));
    let input = Grid::new(2, 1);
    let prediction = target.clone();
    let entry = SampleReportEntry {
        label: "routing <sample>",
        input: &input,
        prediction: &prediction,
        target: &target,
        observed: &[false, true],
        confidence: &[0.8, 1.0],
        entropy: &[0.2, 0.0],
        reveal_step: &[2, 0],
    };

    write_sample_report(&path, &[entry]).unwrap();
    let html = fs::read_to_string(&path).unwrap();
    assert!(html.contains("Confidence"));
    assert!(html.contains("Entropy"));
    assert!(html.contains("Error"));
    assert!(html.contains("Reveal round"));
    assert!(html.contains("routing &lt;sample&gt;"));
    fs::remove_file(path).ok();
}
