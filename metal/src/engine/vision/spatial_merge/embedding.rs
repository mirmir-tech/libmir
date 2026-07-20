use models::{layout::SpatialMergeVisionConfig, vision::SpatialMergePreprocessedImage};

use super::dimension;
use crate::engine::{Array, DenseLinear, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct PatchEmbedding {
    projection: DenseLinear,
    positions: Array,
    patch_layout: PatchProjectionLayout,
    patch_shape: [usize; 4],
    hidden_size: usize,
    grid_side: usize,
    merge: usize,
}

#[derive(Debug, Clone, Copy)]
enum PatchProjectionLayout {
    ChannelFirst,
    ChannelLast,
}

impl PatchEmbedding {
    pub(super) fn load(
        tensors: &ModelTensors,
        config: &SpatialMergeVisionConfig,
        prefix: &str,
        stream: &Stream,
    ) -> Result<Self> {
        let grid_side = integer_sqrt(config.num_position_embeddings);
        if grid_side * grid_side != config.num_position_embeddings {
            return Err(Error::InvalidModel(
                "spatial-merge vision position embedding count must be a square".into(),
            ));
        }
        let projection_prefix = format!("{prefix}.patch_embed.proj");
        let weight = tensors.get(&format!("{projection_prefix}.weight"))?;
        let patch_layout = patch_projection_layout(&weight.shape()?, config)?;
        Ok(Self {
            projection: DenseLinear::load_flattened(tensors, &projection_prefix, stream)?,
            positions: tensors.get(&format!("{prefix}.pos_embed.weight"))?,
            patch_layout,
            patch_shape: [
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ],
            hidden_size: config.hidden_size,
            grid_side,
            merge: config.spatial_merge_size,
        })
    }

    pub(super) fn forward(
        &self,
        patches: &Array,
        image: &SpatialMergePreprocessedImage,
        stream: &Stream,
    ) -> Result<Array> {
        let patches = self.prepare_patches(patches, image, stream)?;
        let hidden = self.projection.forward(&patches, stream)?;
        let positions = self.interpolate(image.grid_height, image.grid_width, stream)?;
        hidden.add(&positions, stream)
    }

    fn prepare_patches(
        &self,
        patches: &Array,
        image: &SpatialMergePreprocessedImage,
        stream: &Stream,
    ) -> Result<Array> {
        if matches!(self.patch_layout, PatchProjectionLayout::ChannelFirst) {
            return Array::from_native(patches.native().clone());
        }
        let sequence =
            image.grid_height.checked_mul(image.grid_width).ok_or(Error::ShapeOverflow)?;
        let [channels, temporal, height, width] = self.patch_shape;
        let [channels, temporal, height, width] = [
            i32::try_from(channels)?,
            i32::try_from(temporal)?,
            i32::try_from(height)?,
            i32::try_from(width)?,
        ];
        patches
            .reshape(
                &[dimension(sequence, "patch sequence")?, channels, temporal, height, width],
                stream,
            )?
            .transpose(&[0, 2, 3, 4, 1], stream)?
            .reshape(
                &[
                    1,
                    dimension(sequence, "patch sequence")?,
                    dimension(self.patch_width()?, "patch width")?,
                ],
                stream,
            )
    }

    fn patch_width(&self) -> Result<usize> {
        self.patch_shape.into_iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(dimension).ok_or(Error::ShapeOverflow)
        })
    }

    fn interpolate(&self, height: usize, width: usize, stream: &Stream) -> Result<Array> {
        if !height.is_multiple_of(self.merge) || !width.is_multiple_of(self.merge) {
            return Err(Error::InvalidModel(
                "spatial-merge vision grid must be divisible by spatial merge size".into(),
            ));
        }
        let sequence = height.checked_mul(width).ok_or(Error::ShapeOverflow)?;
        let mut indices = (0..4).map(|_| Vec::with_capacity(sequence)).collect::<Vec<_>>();
        let mut weights = (0..4).map(|_| Vec::with_capacity(sequence)).collect::<Vec<_>>();
        for block_y in 0..height / self.merge {
            for block_x in 0..width / self.merge {
                for merge_y in 0..self.merge {
                    for merge_x in 0..self.merge {
                        let y = block_y * self.merge + merge_y;
                        let x = block_x * self.merge + merge_x;
                        append_interpolation(
                            &mut indices, &mut weights, y, x, height, width, self.grid_side,
                        )?;
                    }
                }
            }
        }
        let index_values = indices.into_iter().flatten().collect::<Vec<_>>();
        let weight_values = weights.into_iter().flatten().collect::<Vec<_>>();
        let sequence_i32 = dimension(sequence, "position sequence")?;
        let indices = Array::from_u32(&index_values, &[4, sequence_i32])?;
        let weights = Array::from_f32(&weight_values, &[4, sequence_i32, 1])?;
        self.positions
            .take(&indices, 0, stream)?
            .multiply(&weights, stream)?
            .reduce_sum(0, false, stream)?
            .reshape(&[1, sequence_i32, dimension(self.hidden_size, "hidden size")?], stream)
    }
}

fn patch_projection_layout(
    shape: &[i32],
    config: &SpatialMergeVisionConfig,
) -> Result<PatchProjectionLayout> {
    let output = i32::try_from(config.hidden_size)?;
    let channels = i32::try_from(config.in_channels)?;
    let temporal = i32::try_from(config.temporal_patch_size)?;
    let patch = i32::try_from(config.patch_size)?;
    let flattened = channels
        .checked_mul(temporal)
        .and_then(|value| value.checked_mul(patch))
        .and_then(|value| value.checked_mul(patch))
        .ok_or(Error::ShapeOverflow)?;
    match shape {
        [actual_output, width] if *actual_output == output && *width == flattened => {
            Ok(PatchProjectionLayout::ChannelFirst)
        },
        [actual_output, actual_channels, actual_temporal, height, width]
            if [*actual_output, *actual_channels, *actual_temporal, *height, *width]
                == [output, channels, temporal, patch, patch] =>
        {
            Ok(PatchProjectionLayout::ChannelFirst)
        },
        [actual_output, actual_temporal, height, width, actual_channels]
            if [*actual_output, *actual_temporal, *height, *width, *actual_channels]
                == [output, temporal, patch, patch, channels] =>
        {
            Ok(PatchProjectionLayout::ChannelLast)
        },
        _ => Err(Error::InvalidModel(format!(
            "spatial patch projection shape {shape:?} does not match parsed image geometry"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn append_interpolation(
    indices: &mut [Vec<u32>],
    weights: &mut [Vec<f32>],
    y: usize,
    x: usize,
    height: usize,
    width: usize,
    side: usize,
) -> Result<()> {
    let coordinate = |value: usize, size: usize| -> Result<f64> {
        if size == 1 {
            Ok(0.0)
        } else {
            Ok(value.to_string().parse::<f64>()? * (side - 1).to_string().parse::<f64>()?
                / (size - 1).to_string().parse::<f64>()?)
        }
    };
    let y = coordinate(y, height)?;
    let x = coordinate(x, width)?;
    let y0 = float_usize(y.floor())?;
    let x0 = float_usize(x.floor())?;
    let y1 = (y0 + 1).min(side - 1);
    let x1 = (x0 + 1).min(side - 1);
    let dy = y - y0.to_string().parse::<f64>()?;
    let dx = x - x0.to_string().parse::<f64>()?;
    for (slot, index) in [y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1]
        .into_iter()
        .enumerate()
    {
        indices[slot].push(u32::try_from(index)?);
    }
    for (slot, weight) in [(1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx, dy * (1.0 - dx), dy * dx]
        .into_iter()
        .enumerate()
    {
        weights[slot].push(weight.to_string().parse()?);
    }
    Ok(())
}

fn integer_sqrt(value: usize) -> usize {
    let mut root = 0_usize;
    while (root + 1).checked_mul(root + 1).is_some_and(|square| square <= value) {
        root += 1;
    }
    root
}

fn float_usize(value: f64) -> Result<usize> {
    Ok(value.to_string().parse()?)
}
