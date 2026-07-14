//! diffusion-factorio: discrete spatial diffusion for Factorio blueprints.
//!
//! See `docs/DESIGN.md` for the architecture. The crate is split into a pure
//! "world" half (no ML deps) and a "learning" half (burn):
//!
//! * [`world`] — grid / entity representation.
//! * [`sim`] — functional evaluation (does the factory route items?).
//! * [`factory_gen`] — procedural lesson generation + blanking.
//! * [`textual`] — ASCII (de)serialization for debugging & fixtures.
//! * [`backend`] — burn backend selection (CPU ndarray / GPU wgpu).
//! * [`data`] — turn samples into batched tensors.
//! * [`diffusion`] — masked (absorbing-state) discrete diffusion process.
//! * [`model`] — the U-Net denoiser.
//! * [`metrics`] — training / validation metrics.
//! * [`train`] — the training loop.
//! * [`sample`] — inference / sampling & functional validation.

pub mod backend;
pub mod data;
pub mod diffusion;
pub mod factory_gen;
pub mod metrics;
pub mod model;
pub mod persist;
pub mod sample;
pub mod sim;
pub mod textual;
pub mod train;
pub mod world;
