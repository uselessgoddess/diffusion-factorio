# Factorio blueprint export

Generate reconstructions and export the first one:

```bash
cargo run --release --features wgpu --bin sample -- \
  --ckpt checkpoints/denoiser --show 4 --eval 256 \
  --blueprint-out generated-blueprint.txt
```

In Factorio 2.x, open the Blueprint Library (`B`), choose **Import string**, and
paste the contents of the file. The exporter emits the classic exchange format:
version byte `0` followed by base64-encoded zlib JSON.

## Mapping

| Model cell | Factorio prototype |
|---|---|
| Transport belt | `transport-belt` |
| Underground down/up | `underground-belt`, type `input` / `output` |
| Splitter | `splitter` |
| Inserter | `inserter` |
| Assembler | `assembling-machine-1` with the predicted recipe |
| Source / sink | tagged `constant-combinator` marker |

Directions use Factorio 2.x's 16-step enum (`0/4/8/12` for north/east/south/west).
Source and sink are environment concepts rather than placeable factory entities.
The combinator markers retain their role and item in blueprint tags and filters,
so they remain visible and machine-readable after import.

## Current parity boundary

The current learned world is intentionally simplified: assemblers occupy one
model cell and functional scoring is reachability rather than lane/capacity
simulation. A real Factorio assembler is 3×3, so assembler lessons may need
manual spacing after import. Belt and underground routing lessons map directly;
source/sink combinators must be replaced with the desired loaders, chests, or
production machines for a live throughput test.

Blueprint import is therefore the visual and format round trip, not yet proof
of simulator parity. The next trust milestone is to make footprints native in
the world representation, add graded lane-aware throughput, and compare that
simulator against real Factorio ticks before optimizing it with best-of-N or RL.
