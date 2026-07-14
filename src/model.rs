//! The denoiser network `f_theta(x_t, t) -> logits over x_0`.
//!
//! Architecture (see `docs/DESIGN.md`):
//!   * Per-channel token embeddings (Empty/Belt/... and a MASK id). Category ids
//!     are embedded, never fed as raw ordinals — same rationale as the
//!     reference's `_encode_input`.
//!   * A residual conv tower operating on the full grid, each block injecting a
//!     **global-context vector** (mean-pool -> linear -> broadcast) and a
//!     **time-conditioning vector** (FiLM-style add). The global context
//!     directly addresses the reference's receptive-field bottleneck ("which way
//!     is the sink" is a grid-global question a local window can't answer).
//!   * One 1x1 conv output head per channel, predicting real classes only
//!     (never MASK) -> jointly denoises all coupled channels.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Gelu, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::gelu;

use crate::world::{N_CHANNELS, VOCAB};

#[derive(Config, Debug)]
pub struct DenoiserConfig {
    #[config(default = 16)]
    pub embed_dim: usize,
    #[config(default = 64)]
    pub hidden: usize,
    #[config(default = 6)]
    pub blocks: usize,
    #[config(default = 64)]
    pub time_dim: usize,
}

impl DenoiserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Denoiser<B> {
        // One embedding per channel; +1 row for the MASK/absorbing token.
        let embeds: Vec<Embedding<B>> = (0..N_CHANNELS)
            .map(|c| EmbeddingConfig::new(VOCAB[c] + 1, self.embed_dim).init(device))
            .collect();

        let in_ch = N_CHANNELS * self.embed_dim + 1; // + obstacle conditioning
        let stem = Conv2dConfig::new([in_ch, self.hidden], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        // Sinusoidal time embedding -> hidden vector (added into every block).
        let time_mlp1 = LinearConfig::new(self.time_dim, self.hidden).init(device);
        let time_mlp2 = LinearConfig::new(self.hidden, self.hidden).init(device);

        let blocks = (0..self.blocks)
            .map(|_| ResBlock::new(self.hidden, device))
            .collect();

        let heads: Vec<Conv2d<B>> = (0..N_CHANNELS)
            .map(|c| Conv2dConfig::new([self.hidden, VOCAB[c]], [1, 1]).init(device))
            .collect();

        Denoiser {
            embeds,
            stem,
            time_mlp1,
            time_mlp2,
            blocks,
            heads,
            activation: Gelu::new(),
            time_dim: self.time_dim,
            hidden: self.hidden,
        }
    }
}

#[derive(Module, Debug)]
pub struct Denoiser<B: Backend> {
    embeds: Vec<Embedding<B>>,
    stem: Conv2d<B>,
    time_mlp1: Linear<B>,
    time_mlp2: Linear<B>,
    blocks: Vec<ResBlock<B>>,
    heads: Vec<Conv2d<B>>,
    activation: Gelu,
    time_dim: usize,
    hidden: usize,
}

impl<B: Backend> Denoiser<B> {
    /// `tokens`: `[batch, N_CHANNELS, H, W]` int (may include MASK id).
    /// `obstacle`: `[batch, 1, H, W]` float in {0,1}.
    /// `t`: `[batch]` float in (0,1].
    /// Returns one logits tensor per channel: `[batch, VOCAB[c], H, W]`.
    pub fn forward(
        &self,
        tokens: Tensor<B, 4, Int>,
        obstacle: Tensor<B, 4>,
        t: Tensor<B, 1>,
    ) -> Vec<Tensor<B, 4>> {
        let [batch, _c, height, width] = tokens.dims();
        let device = tokens.device();

        // Embed each channel and stack along the feature axis.
        let mut feats: Vec<Tensor<B, 4>> = Vec::with_capacity(N_CHANNELS);
        for c in 0..N_CHANNELS {
            let ch = tokens
                .clone()
                .slice([0..batch, c..c + 1, 0..height, 0..width]); // [B,1,H,W]
            let ch = ch.reshape([batch, height * width]); // [B, H*W] Int
            let emb = self.embeds[c].forward(ch); // [B, H*W, E]
            let e = emb.dims()[2];
            let emb = emb.reshape([batch, height, width, e]).permute([0, 3, 1, 2]); // [B, E, H, W]
            feats.push(emb);
        }
        feats.push(obstacle); // [B, 1, H, W]
        let x = Tensor::cat(feats, 1); // [B, N*E+1, H, W]

        let mut h = self.stem.forward(x); // [B, hidden, H, W]

        // Time conditioning: sinusoidal(t) -> MLP -> [B, hidden] -> broadcast.
        let temb = sinusoidal_time_embedding(t, self.time_dim, &device); // [B, time_dim]
        let temb = self.time_mlp2.forward(gelu(self.time_mlp1.forward(temb))); // [B, hidden]
        let temb = temb.reshape([batch, self.hidden, 1, 1]);

        for block in &self.blocks {
            h = block.forward(h, temb.clone());
        }

        self.heads
            .iter()
            .map(|head| head.forward(h.clone()))
            .collect()
    }
}

/// Residual conv block with global-context and time injection.
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    global: Linear<B>,
}

impl<B: Backend> ResBlock<B> {
    fn new(hidden: usize, device: &B::Device) -> Self {
        let conv = || {
            Conv2dConfig::new([hidden, hidden], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv1: conv(),
            conv2: conv(),
            global: LinearConfig::new(hidden, hidden).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>, temb: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, ch, _h, _w] = x.dims();
        // Global context: mean over spatial dims -> linear -> broadcast add.
        let g = x.clone().mean_dim(3).mean_dim(2); // [B, C, 1, 1]
        let g = g.reshape([batch, ch]);
        let g = self.global.forward(g).reshape([batch, ch, 1, 1]);

        let mut h = self.conv1.forward(x.clone());
        h = gelu(h + temb + g);
        h = self.conv2.forward(h);
        x + h
    }
}

/// Standard sinusoidal embedding of a continuous scalar `t in (0,1]`.
fn sinusoidal_time_embedding<B: Backend>(
    t: Tensor<B, 1>,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch = t.dims()[0];
    let half = dim / 2;
    // freqs[i] = exp(-ln(10000) * i / half)
    let idx = Tensor::<B, 1, Int>::arange(0..half as i64, device).float();
    let freqs = idx.mul_scalar(-(10000f64.ln()) / half as f64).exp(); // [half]
    let t = t.reshape([batch, 1]); // [B,1]
    let freqs = freqs.reshape([1, half]); // [1,half]
    let args = t.mul(freqs); // [B, half]
    let emb = Tensor::cat(vec![args.clone().sin(), args.cos()], 1); // [B, 2*half]
                                                                    // Pad if dim is odd.
    if 2 * half == dim {
        emb
    } else {
        Tensor::cat(vec![emb, Tensor::zeros([batch, dim - 2 * half], device)], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn forward_shapes_are_correct() {
        type B = CpuBackend;
        let device = Default::default();
        let model = DenoiserConfig::new()
            .with_hidden(16)
            .with_blocks(2)
            .init::<B>(&device);
        let (batch, h, w) = (2, 11, 11);
        let tokens = Tensor::<B, 4, Int>::zeros([batch, N_CHANNELS, h, w], &device);
        let obstacle = Tensor::<B, 4>::zeros([batch, 1, h, w], &device);
        let t = Tensor::<B, 1>::from_floats([0.5, 0.9], &device);
        let out = model.forward(tokens, obstacle, t);
        assert_eq!(out.len(), N_CHANNELS);
        for (c, logits) in out.iter().enumerate() {
            assert_eq!(logits.dims(), [batch, VOCAB[c], h, w]);
        }
    }
}
