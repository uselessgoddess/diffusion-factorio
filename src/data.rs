//! Turn factory grids into batched burn tensors and back.

use burn::prelude::*;
use burn::tensor::{Bool, Int, TensorData};

use crate::world::{Cell, Channel, Direction, Entity, Grid, Item, Misc, N_CHANNELS};

const CHANNELS: [Channel; N_CHANNELS] = [
    Channel::Entity,
    Channel::Direction,
    Channel::Item,
    Channel::Misc,
];

/// A batch of grids as tensors.
///
/// * `tokens`: `[n, N_CHANNELS, H, W]` int category ids (no MASK; ground truth).
/// * `obstacle`: `[n, 1, H, W]` float in {0,1} (conditioning).
/// * `observed`: `[n, H, W]` bool — `true` = a given/conditioning cell that must
///   never be masked (used for conditional generation / inpainting).
#[derive(Clone, Debug)]
pub struct GridBatch<B: Backend> {
    pub tokens: Tensor<B, 4, Int>,
    pub obstacle: Tensor<B, 4>,
    pub observed: Tensor<B, 3, Bool>,
    pub height: usize,
    pub width: usize,
}

impl<B: Backend> GridBatch<B> {
    pub fn batch_size(&self) -> usize {
        self.tokens.dims()[0]
    }

    /// Build a batch from grids. All grids must share dimensions. `observed`, if
    /// given, marks conditioning cells per grid (defaults to all-observed=false
    /// -> fully generative, which is what unconditional training wants).
    pub fn from_grids(grids: &[Grid], observed: Option<&[Vec<bool>]>, device: &B::Device) -> Self {
        assert!(!grids.is_empty(), "empty batch");
        let (h, w) = (grids[0].height, grids[0].width);
        let n = grids.len();

        let mut tokens = Vec::with_capacity(n * N_CHANNELS * h * w);
        let mut obstacle = Vec::with_capacity(n * h * w);
        let mut obs = Vec::with_capacity(n * h * w);

        for (gi, g) in grids.iter().enumerate() {
            assert_eq!((g.height, g.width), (h, w), "ragged batch");
            for (ci, ch) in CHANNELS.iter().enumerate() {
                let _ = ci;
                for y in 0..h {
                    for x in 0..w {
                        tokens.push(g.get(x, y).channel_id(*ch) as i32);
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    obstacle.push(if g.is_obstacle(x, y) { 1.0f32 } else { 0.0 });
                    let o = observed.map(|m| m[gi][g.idx(x, y)]).unwrap_or(false);
                    obs.push(o);
                }
            }
        }

        let tokens =
            Tensor::<B, 4, Int>::from_data(TensorData::new(tokens, [n, N_CHANNELS, h, w]), device);
        let obstacle = Tensor::<B, 4>::from_data(TensorData::new(obstacle, [n, 1, h, w]), device);
        let observed = Tensor::<B, 3, Bool>::from_data(TensorData::new(obs, [n, h, w]), device);

        Self {
            tokens,
            obstacle,
            observed,
            height: h,
            width: w,
        }
    }
}

/// Decode a `[N_CHANNELS, H, W]` slice of category ids back into a [`Grid`].
/// Values are clamped/validated; out-of-range ids fall back to the 0 class.
pub fn grid_from_ids(ids: &[i32], height: usize, width: usize, obstacle: &[bool]) -> Grid {
    let mut g = Grid::new(width, height);
    let plane = height * width;
    for y in 0..height {
        for x in 0..width {
            let at = |c: usize| ids[c * plane + y * width + x].max(0) as usize;
            let cell = Cell {
                entity: Entity::from_id(at(0)).unwrap_or(Entity::Empty),
                direction: Direction::from_id(at(1)).unwrap_or(Direction::None),
                item: Item::from_id(at(2)).unwrap_or(Item::None),
                misc: Misc::from_id(at(3)).unwrap_or(Misc::None),
            };
            g.set(x, y, cell);
            g.set_obstacle(x, y, obstacle[y * width + x]);
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::factory_gen::{generate, Canvas, LessonKind};

    #[test]
    fn roundtrip_grid_through_tensor() {
        type B = CpuBackend;
        let device = Default::default();
        let s = generate(LessonKind::MoveOneItem, Canvas::square(11), 3).unwrap();
        let batch = GridBatch::<B>::from_grids(std::slice::from_ref(&s.solution), None, &device);

        let data = batch.tokens.to_data();
        let ids: Vec<i32> = data.convert::<i32>().into_vec().unwrap();
        let obstacle: Vec<bool> = (0..s.solution.len())
            .map(|i| s.solution.obstacle[i])
            .collect();
        let decoded = grid_from_ids(&ids, 11, 11, &obstacle);
        assert_eq!(decoded, s.solution);
    }
}
