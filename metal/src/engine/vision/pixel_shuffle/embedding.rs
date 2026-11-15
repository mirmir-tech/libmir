use models::layout::PixelShuffleVisionConfig;

use super::{dimension, encoder};
use crate::engine::{Array, DenseLinear, Error, ModelTensors, Result, Stream};

/// Prefix tokens, then each projected patch plus its position: the square
/// table resized bilinearly (half-pixel centres, clamped at the first texel)
/// to the larger grid side and read at the patch.
#[derive(Debug)]
pub(super) struct PatchEmbedding {
    projection: DenseLinear,
    positions: Array,
    prefix: Array,
    grid: usize,
    hidden: usize,
}

impl PatchEmbedding {
    pub(super) fn load(
        tensors: &ModelTensors,
        config: &PixelShuffleVisionConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let hidden = dimension(config.hidden_size, "hidden size")?;
        let grid = config.position_grid;
        let cells = dimension(grid.checked_mul(grid).ok_or(Error::ShapeOverflow)?, "positions")?;
        Ok(Self {
            projection: DenseLinear::load(tensors, &encoder("patch_generator.embedder"), stream)?,
            positions: tensors
                .get(&encoder("patch_generator.pos_embed"))?
                .reshape(&[cells, hidden], stream)?,
            prefix: tensors
                .get(&encoder("patch_generator.cls_token.token"))?
                .reshape(&[1, dimension(config.prefix_tokens, "prefix tokens")?, hidden], stream)?,
            grid,
            hidden: config.hidden_size,
        })
    }

    pub(super) fn forward(
        &self,
        patches: &Array,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let projected = self.projection.forward(patches, stream)?;
        let positions =
            self.resized_positions(rows, cols, stream)?.astype_like(&projected, stream)?;
        let patches = projected.add(&positions, stream)?;
        Array::concatenate(&[&self.prefix.astype_like(&patches, stream)?, &patches], 1, stream)
    }

    fn resized_positions(&self, rows: usize, cols: usize, stream: &Stream) -> Result<Array> {
        let count = rows.checked_mul(cols).ok_or(Error::ShapeOverflow)?;
        let side = rows.max(cols);
        let scale = real(self.grid)? / real(side)?;
        let mut indices = (0..4).map(|_| Vec::with_capacity(count)).collect::<Vec<_>>();
        let mut weights = (0..4).map(|_| Vec::with_capacity(count)).collect::<Vec<_>>();
        for row in 0..rows {
            let (y0, y1, dy) = texels(row, scale, self.grid)?;
            for col in 0..cols {
                let (x0, x1, dx) = texels(col, scale, self.grid)?;
                let corners = [
                    (y0 * self.grid + x0, (1.0 - dy) * (1.0 - dx)),
                    (y0 * self.grid + x1, (1.0 - dy) * dx),
                    (y1 * self.grid + x0, dy * (1.0 - dx)),
                    (y1 * self.grid + x1, dy * dx),
                ];
                for (slot, (index, weight)) in corners.into_iter().enumerate() {
                    indices[slot].push(u32::try_from(index)?);
                    weights[slot].push(weight);
                }
            }
        }
        let count = dimension(count, "patch count")?;
        let indices = Array::from_u32(&indices.concat(), &[4, count])?;
        let weights = Array::from_f32(&weights.concat(), &[4, count, 1])?;
        self.positions
            .take(&indices, 0, stream)?
            .astype(crate::engine::Dtype::Float32, stream)?
            .multiply(&weights, stream)?
            .reduce_sum(0, false, stream)?
            .reshape(&[1, count, dimension(self.hidden, "hidden size")?], stream)
    }
}

/// The two table texels a patch coordinate falls between and its weight.
fn texels(index: usize, scale: f32, grid: usize) -> Result<(usize, usize, f32)> {
    let source = (real(index)? + 0.5).mul_add(scale, -0.5).max(0.0);
    let lower = (0..grid).rev().find(|texel| real(*texel).is_ok_and(|texel| texel <= source));
    let lower = lower
        .ok_or_else(|| Error::InvalidModel("pixel-shuffle vision position grid is empty".into()))?;
    Ok((lower, (lower + 1).min(grid - 1), source - real(lower)?))
}

fn real(value: usize) -> Result<f32> {
    let value = u16::try_from(value).map_err(|_| {
        Error::InvalidModel("pixel-shuffle vision grid exceeds the position table range".into())
    })?;
    Ok(f32::from(value))
}
