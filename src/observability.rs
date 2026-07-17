//! Durable, dependency-free observability artifacts for training and sampling.
//!
//! Metrics are JSONL so they remain usable by notebooks and experiment tools.
//! Reports are a single offline HTML file with embedded data and canvas charts,
//! making a long run inspectable without an account or network connection.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::metrics::ReconReport;
use crate::train::TrainLog;
use crate::world::Grid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub backend: String,
    /// The canvas shapes the run trained on, e.g. `9x9 .. 15x15` — a pool, not a
    /// number, since a run no longer sees one shape.
    pub canvases: String,
    pub steps: usize,
    pub batch_size: usize,
    pub val_batch: usize,
    pub sample_steps: usize,
    pub seed: u64,
    pub legacy_protected_scaffold: bool,
    pub peak_lr: f64,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub hidden: usize,
    pub blocks: usize,
    pub embed_dim: usize,
    pub time_dim: usize,
    pub elbo_weight: bool,
    pub t_min: f64,
    pub scratch_probability: f64,
    pub structure_weight: f64,
}

/// Streaming JSONL writer. Each append is flushed so interrupted GPU runs keep
/// all completed telemetry records.
pub struct MetricsWriter {
    writer: BufWriter<File>,
}

impl MetricsWriter {
    pub fn create(path: &Path) -> Result<Self> {
        ensure_parent(path)?;
        let file = File::create(path)
            .with_context(|| format!("create metrics file {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn append(&mut self, log: &TrainLog) -> Result<()> {
        serde_json::to_writer(&mut self.writer, &metric_record(log))
            .context("serialize training metric")?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

pub fn write_metrics_jsonl(path: &Path, logs: &[TrainLog]) -> Result<()> {
    let mut writer = MetricsWriter::create(path)?;
    for log in logs {
        writer.append(log)?;
    }
    Ok(())
}

/// Write an offline report containing parameters, definitions and interactive
/// curves. It intentionally has no CDN dependencies.
pub fn write_training_report(path: &Path, metadata: &RunMetadata, logs: &[TrainLog]) -> Result<()> {
    ensure_parent(path)?;
    let payload = json!({
        "metadata": metadata,
        "metrics": logs.iter().map(metric_record).collect::<Vec<_>>()
    });
    let payload = serde_json::to_string(&payload)
        .context("serialize embedded report data")?
        .replace("</", "<\\/");
    let html = TRAINING_REPORT_TEMPLATE.replace("__REPORT_DATA__", &payload);
    fs::write(path, html).with_context(|| format!("write report {}", path.display()))
}

/// Inputs for one reconstruction card in the spatial diagnostics report.
pub struct SampleReportEntry<'a> {
    pub label: &'a str,
    pub input: &'a Grid,
    pub prediction: &'a Grid,
    pub target: &'a Grid,
    pub observed: &'a [bool],
    pub confidence: &'a [f32],
    pub entropy: &'a [f32],
    pub reveal_step: &'a [usize],
}

/// Write input/prediction/target grids beside confidence, entropy, error and
/// reveal-round heatmaps. Values come from the moment each cell is committed,
/// rather than a leakage-prone forward pass over the finished answer.
pub fn write_sample_report(path: &Path, entries: &[SampleReportEntry<'_>]) -> Result<()> {
    ensure_parent(path)?;
    let mut cards = String::new();
    for entry in entries {
        let n = entry.input.len();
        anyhow::ensure!(
            entry.prediction.len() == n && entry.target.len() == n,
            "ragged report grid"
        );
        anyhow::ensure!(
            entry.observed.len() == n
                && entry.confidence.len() == n
                && entry.entropy.len() == n
                && entry.reveal_step.len() == n,
            "diagnostic map length does not match grid"
        );
        let errors: Vec<f32> = (0..n)
            .map(|i| {
                if !entry.observed[i] && entry.prediction.cells[i] != entry.target.cells[i] {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let max_step = entry.reveal_step.iter().copied().max().unwrap_or(1).max(1) as f32;
        let reveal: Vec<f32> = entry
            .reveal_step
            .iter()
            .map(|&step| step as f32 / max_step)
            .collect();
        cards.push_str(&format!(
            "<section class=\"sample\"><h2>{}</h2><div class=\"panels\">{}{}{}</div>\
             <div class=\"panels\">{}{}{}{}{}{}{}</div></section>",
            escape_html(entry.label),
            svg_panel("Given", entry.input),
            svg_panel("Model's factory", entry.prediction),
            svg_panel("Ground truth", entry.target),
            grid_panel("Masked input", entry.input),
            grid_panel("Prediction", entry.prediction),
            grid_panel("Ground truth", entry.target),
            heatmap_panel("Confidence", entry.input.width, entry.confidence, false),
            heatmap_panel("Entropy", entry.input.width, entry.entropy, true),
            heatmap_panel("Error", entry.input.width, &errors, true),
            heatmap_panel("Reveal round", entry.input.width, &reveal, false),
        ));
    }
    let html = SAMPLE_REPORT_TEMPLATE.replace("__SAMPLE_CARDS__", &cards);
    fs::write(path, html).with_context(|| format!("write sample report {}", path.display()))
}

fn metric_record(log: &TrainLog) -> Value {
    let val = log.val.as_ref().map(validation_record);
    let by_lesson = |reports: &BTreeMap<String, ReconReport>| -> serde_json::Map<String, Value> {
        reports
            .iter()
            .map(|(name, report)| (name.clone(), validation_record(report)))
            .collect()
    };
    json!({
        "step": log.step,
        "elapsed_seconds": log.elapsed_seconds,
        "samples_seen": log.samples_seen,
        "samples_per_second": log.samples_per_second,
        "lr": log.lr,
        "loss": log.loss,
        "nll": log.nll,
        "mask_rate": log.t_mean,
        "placement_recall": log.placement_acc,
        "assembler_recall": log.assembler_acc,
        "recipe_accuracy": log.recipe_acc,
        "train": {
            "entity_acc": log.train_acc[0],
            "direction_acc": log.train_acc[1],
            "item_acc": log.train_acc[2],
            "misc_acc": log.train_acc[3],
            "entity_nll": log.channel_nll[0],
            "direction_nll": log.channel_nll[1],
            "item_nll": log.channel_nll[2],
            "misc_nll": log.channel_nll[3]
        },
        "val": val,
        "val_by_lesson": by_lesson(&log.val_by_lesson),
        "val_scratch": log.val_scratch.as_ref().map(validation_record),
        "val_scratch_by_lesson": by_lesson(&log.val_scratch_by_lesson)
    })
}

fn validation_record(r: &ReconReport) -> Value {
    json!({
        "n": r.n_factories,
        "masked_cells": r.masked_cells,
        "exact_rate": r.exact_rate(),
        "functional_rate": r.functional_rate(),
        "consistent_rate": r.consistent_rate(),
        "original_functional": r.original_functional,
        "throughput": r.throughput_mean(),
        "original_throughput": r.original_throughput_mean(),
        "throughput_ratio": r.throughput_ratio_mean(),
        "beat_original": r.beat_original,
        "entity_acc": r.channel_acc(0),
        "direction_acc": r.channel_acc(1),
        "item_acc": r.channel_acc(2),
        "misc_acc": r.channel_acc(3)
    })
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    Ok(())
}

/// The picture of a factory, as opposed to [`grid_panel`]'s picture of the
/// tensor. Machines appear at the size Factorio gives them, so a reader can see
/// a 3×3 assembler with inserters on its edge rather than decode `A` and `a`.
fn svg_panel(title: &str, grid: &Grid) -> String {
    format!(
        "<div><h3>{title}</h3>{}</div>",
        crate::viewer::grid_to_svg(grid)
    )
}

fn grid_panel(title: &str, grid: &Grid) -> String {
    let rendered = crate::textual::render(grid);
    let cells: String = rendered
        .chars()
        .filter(|&ch| ch != '\n' && ch != '\r')
        .map(|ch| format!("<i>{}</i>", escape_html(&ch.to_string())))
        .collect();
    format!(
        "<div><h3>{title}</h3><div class=\"grid glyphs\" style=\"--w:{}\">{cells}</div></div>",
        grid.width
    )
}

fn heatmap_panel(title: &str, width: usize, values: &[f32], danger: bool) -> String {
    let cells: String = values
        .iter()
        .map(|&raw| {
            let value = raw.clamp(0.0, 1.0);
            let hue = if danger {
                120.0 * (1.0 - value)
            } else {
                120.0 * value
            };
            format!("<i title=\"{value:.3}\" style=\"background:hsl({hue:.1} 72% 42%)\"></i>")
        })
        .collect();
    format!("<div><h3>{title}</h3><div class=\"grid\" style=\"--w:{width}\">{cells}</div></div>")
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const TRAINING_REPORT_TEMPLATE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>diffusion-factorio training report</title>
<style>
:root{color-scheme:dark;--bg:#101418;--panel:#182027;--ink:#e8edf2;--muted:#9eabb6;--grid:#34414c;--a:#5ee6a8;--b:#ffcc66;--c:#79b8ff;--d:#ff7b9c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 5px}h2{font-size:17px;margin:0 0 12px}.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:22px 0}.card,.chart{background:var(--panel);border:1px solid #26333d;border-radius:10px;padding:15px}.card b{display:block;font-size:20px;margin-top:3px}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}.chart canvas{width:100%;height:250px}table{width:100%;border-collapse:collapse}td{padding:7px;border-bottom:1px solid #27343e;vertical-align:top}td:first-child{font-family:ui-monospace,monospace;color:var(--a);white-space:nowrap}.definitions{margin-top:16px}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}
@media(max-width:600px){main{padding:16px}.charts{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Training report</h1><div class="muted">Frozen validation · durable JSONL metrics · offline report</div>
<section class="cards" id="summary"></section>
<section class="charts">
 <div class="chart"><h2>Loss and per-channel NLL</h2><canvas id="loss"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>weighted loss</span><span><i class="dot" style="background:var(--b)"></i>entity NLL</span><span><i class="dot" style="background:var(--c)"></i>direction NLL</span></div></div>
 <div class="chart"><h2>Learning rate</h2><canvas id="schedule"></canvas><div class="legend"><span><i class="dot" style="background:var(--c)"></i>warmup + cosine schedule</span></div></div>
 <div class="chart"><h2>Training throughput</h2><canvas id="speed"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>average samples / second</span></div></div>
 <div class="chart"><h2>Placement and recipe learning</h2><canvas id="train"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>Assembler recall</span><span><i class="dot" style="background:var(--c)"></i>Recipe accuracy</span><span><i class="dot" style="background:var(--b)"></i>All non-empty placement</span></div></div>
 <div class="chart"><h2>Functional / exact / consistent</h2><canvas id="validation"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>functional</span><span><i class="dot" style="background:var(--c)"></i>exact</span><span><i class="dot" style="background:var(--b)"></i>consistent</span></div></div>
 <div class="chart"><h2>Built from scratch (source and sink only)</h2><canvas id="scratch"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>functional</span><span><i class="dot" style="background:var(--c)"></i>exact</span></div></div>
 <div class="chart"><h2>Delivered throughput vs. the taught answer</h2><canvas id="rate"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>in-painting</span><span><i class="dot" style="background:var(--c)"></i>from scratch</span><span><i class="dot" style="background:var(--d)"></i>parity with the generator</span></div><div class="muted">Items/second delivered as a fraction of what the generator's own answer delivers. This is the only curve that can separate two factories that both work — and the only one that can go <b>above</b> 1.0, which means the model out-built what it was taught.</div></div>
 <div class="chart"><h2>Delivered items / second</h2><canvas id="thput"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>in-painting</span><span><i class="dot" style="background:var(--c)"></i>from scratch</span><span><i class="dot" style="background:var(--b)"></i>the taught answer</span></div><div class="muted">The same quantity the ratio above normalizes, in absolute vanilla items/second. The ratio hides two things this shows: whether a rising ratio came from the model improving or from an easier draw of tasks (the taught-answer curve moves too), and how much of a belt the factory actually fills.</div></div>
 <div class="chart"><h2>Factories that out-built the taught answer</h2><canvas id="beat"></canvas><div class="legend"><span><i class="dot" style="background:var(--a)"></i>in-painting</span><span><i class="dot" style="background:var(--c)"></i>from scratch</span></div><div class="muted">Fraction of reconstructions delivering <b>more</b> items/second than the answer the generator taught. Only an ambiguous lesson family can move this — where a task admits several working answers, this counts the times the model picked a better one than the draw it was shown.</div></div>
</section>
<section class="card definitions"><h2>Run parameters and what they control</h2><table id="parameters"></table></section>
<section class="card definitions"><h2>Latest frozen validation by lesson</h2><table id="lessons"></table></section>
<section class="card definitions"><h2>Latest from-scratch validation by lesson</h2><div class="muted">Everything except the source and sink is masked, so the model designs the factory instead of filling gaps. Read <b>functional</b>: many layouts deliver the item, so <b>exact</b> only rewards rediscovering the generator's own answer.</div><table id="scratch-lessons"></table></section>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent),m=data.metadata,rows=data.metrics;
const last=rows[rows.length-1]||{},lastVal=[...rows].reverse().find(x=>x.val)?.val,lastScratch=[...rows].reverse().find(x=>x.val_scratch)?.val_scratch;
const pct=v=>v===undefined?'—':(100*v).toFixed(1)+'%';
const cards=[['Backend',m.backend],['Canvases',m.canvases],['Samples seen',(last.samples_seen||0).toLocaleString()],['Samples / sec',(last.samples_per_second||0).toFixed(1)],['Final loss',(last.loss||0).toFixed(4)],['Validation functional',pct(lastVal?.functional_rate)],['From-scratch functional',pct(lastScratch?.functional_rate)],['From-scratch throughput',pct(lastScratch?.throughput_ratio)+' of taught'],['Beat the taught answer',(lastScratch?.beat_original??'—')+' / '+(lastScratch?.n??'—')]];
document.getElementById('summary').innerHTML=cards.map(x=>`<div class="card"><span class="muted">${x[0]}</span><b>${x[1]}</b></div>`).join('');
const meanings={backend:'Compute backend used for this run.',canvases:'The canvas shapes the curriculum drew from; one shape per batch.',steps:'Optimizer updates.',batch_size:'Fresh procedural factories per update.',val_batch:'Fixed held-out factories scored at each validation.',sample_steps:'Reverse-diffusion reveal rounds during validation.',seed:'Controls training and the independent frozen validation corpus.',legacy_protected_scaffold:'A/B control that reveals protected answer cells; false for real training.',peak_lr:'Maximum AdamW learning rate.',warmup_steps:'Linear ramp before cosine decay.',grad_clip:'Maximum gradient norm.',hidden:'Convolution tower width.',blocks:'Residual convolution blocks.',embed_dim:'Embedding width per categorical channel.',time_dim:'Diffusion-time embedding width.',elbo_weight:'Use continuous-time 1/t ELBO weighting.',t_min:'Minimum diffusion time used by ELBO weighting.',scratch_probability:'Fraction of examples noised to the exact fully masked state used to begin sampling.',structure_weight:'Extra loss weight for non-empty target cells; counters empty collapse.'};
document.getElementById('parameters').innerHTML=Object.entries(m).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td><td class="muted">${meanings[k]||''}</td></tr>`).join('');
function lessonTable(id,key){const latest=[...rows].reverse().find(r=>Object.keys(r[key]||{}).length)?.[key]||{};document.getElementById(id).innerHTML='<tr><td>lesson</td><td>n</td><td>functional</td><td>exact</td><td>consistent</td><td>throughput</td><td>beat</td></tr>'+Object.entries(latest).map(([name,v])=>`<tr><td>${name}</td><td>${v.n}</td><td>${pct(v.functional_rate)}</td><td>${pct(v.exact_rate)}</td><td>${pct(v.consistent_rate)}</td><td>${pct(v.throughput_ratio)}</td><td>${v.beat_original??'—'}</td></tr>`).join('');}
lessonTable('lessons','val_by_lesson');lessonTable('scratch-lessons','val_scratch_by_lesson');
function chart(id,series,yFixed=false){const c=document.getElementById(id),ctx=c.getContext('2d'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.strokeStyle='#34414c';ctx.lineWidth=1;for(let i=0;i<5;i++){let y=12+i*(h-30)/4;ctx.beginPath();ctx.moveTo(42,y);ctx.lineTo(w-8,y);ctx.stroke()}const vals=series.flatMap(s=>s.values.map(x=>x[1])).filter(Number.isFinite),max=yFixed?1:vals.reduce((a,v)=>Math.max(a,v),1e-12),min=yFixed?0:vals.reduce((a,v)=>Math.min(a,v),0);for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();let started=false;for(const [step,v] of s.values){if(!Number.isFinite(v))continue;const x=42+(w-52)*(step-1)/Math.max(1,m.steps-1),y=12+(h-30)*(1-(v-min)/Math.max(1e-12,max-min));started?ctx.lineTo(x,y):ctx.moveTo(x,y);started=true}ctx.stroke()}ctx.fillStyle='#9eabb6';ctx.font='11px system-ui';ctx.fillText(max.toPrecision(3),3,14);ctx.fillText(min.toPrecision(3),3,h-8)}
const xy=f=>rows.map(r=>[r.step,f(r)]);chart('loss',[{color:'#5ee6a8',values:xy(r=>r.loss)},{color:'#ffcc66',values:xy(r=>r.train.entity_nll)},{color:'#79b8ff',values:xy(r=>r.train.direction_nll)}]);chart('schedule',[{color:'#79b8ff',values:xy(r=>r.lr)}]);chart('speed',[{color:'#5ee6a8',values:xy(r=>r.samples_per_second)}]);chart('train',[{color:'#5ee6a8',values:xy(r=>r.assembler_recall)},{color:'#79b8ff',values:xy(r=>r.recipe_accuracy)},{color:'#ffcc66',values:xy(r=>r.placement_recall)}],true);const vr=rows.filter(r=>r.val);chart('validation',[{color:'#5ee6a8',values:vr.map(r=>[r.step,r.val.functional_rate])},{color:'#79b8ff',values:vr.map(r=>[r.step,r.val.exact_rate])},{color:'#ffcc66',values:vr.map(r=>[r.step,r.val.consistent_rate])}],true);const sr=rows.filter(r=>r.val_scratch);chart('scratch',[{color:'#5ee6a8',values:sr.map(r=>[r.step,r.val_scratch.functional_rate])},{color:'#79b8ff',values:sr.map(r=>[r.step,r.val_scratch.exact_rate])}],true);chart('rate',[{color:'#5ee6a8',values:vr.map(r=>[r.step,r.val.throughput_ratio])},{color:'#79b8ff',values:sr.map(r=>[r.step,r.val_scratch.throughput_ratio])},{color:'#ff7b9c',values:rows.map(r=>[r.step,1])}]);
chart('thput',[{color:'#5ee6a8',values:vr.map(r=>[r.step,r.val.throughput])},{color:'#79b8ff',values:sr.map(r=>[r.step,r.val_scratch.throughput])},{color:'#ffcc66',values:vr.map(r=>[r.step,r.val.original_throughput])}]);
const frac=(v,k)=>v.n?v[k]/v.n:0;chart('beat',[{color:'#5ee6a8',values:vr.map(r=>[r.step,frac(r.val,'beat_original')])},{color:'#79b8ff',values:sr.map(r=>[r.step,frac(r.val_scratch,'beat_original')])}],true);
</script></main></body></html>"#;

const SAMPLE_REPORT_TEMPLATE: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>diffusion-factorio reconstruction diagnostics</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101418;color:#e8edf2;font:14px/1.4 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{margin:0}.intro{color:#9eabb6;margin:5px 0 22px}.sample{background:#182027;border:1px solid #293740;border-radius:10px;padding:16px;margin:14px 0}.sample h2{margin:0 0 12px}.panels{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin-bottom:16px}.panels:last-child{margin-bottom:0}.panels h3{font-size:12px;color:#aeb9c2;margin:0 0 6px}.factory{display:block;width:330px;max-width:100%;height:auto;border-radius:5px}.grid{display:grid;grid-template-columns:repeat(var(--w),18px);grid-auto-rows:18px;gap:1px;background:#0b0e11;padding:4px;border-radius:4px}.grid i{display:block;min-width:18px;min-height:18px}.glyphs i{font:14px/18px ui-monospace,monospace;text-align:center;background:#26313a;font-style:normal}
</style></head><body><main><h1>Reconstruction diagnostics</h1><div class="intro">Confidence and normalized entropy are captured when each cell is revealed. Error excludes observed conditioning cells.</div>__SAMPLE_CARDS__</main></body></html>"#;
