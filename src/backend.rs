//! Backend selection. CPU (ndarray) is always available; wgpu is opt-in behind
//! the `wgpu` feature for GPU training on the user's rx 9070 xt.

use burn::backend::Autodiff;

/// Inference backend (no autodiff) for the CPU path.
pub type CpuBackend = burn::backend::NdArray<f32>;
/// Training backend (autodiff) for the CPU path — used by CI and unit tests.
pub type CpuAutodiff = Autodiff<CpuBackend>;

#[cfg(feature = "wgpu")]
pub type GpuBackend = burn::backend::Wgpu<f32, i32>;
#[cfg(feature = "wgpu")]
pub type GpuAutodiff = Autodiff<GpuBackend>;
