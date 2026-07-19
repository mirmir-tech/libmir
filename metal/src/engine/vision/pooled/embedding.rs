use super::slice_axis;
use crate::engine::{Array, DenseLinear, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct PatchEmbedding {
    projection: DenseLinear,
    position_table: Array,
}

impl PatchEmbedding {
    pub(super) fn load(tensors: &ModelTensors, stream: &Stream) -> Result<Self> {
        Ok(Self {
            projection: DenseLinear::load(
                tensors,
                "model.vision_tower.patch_embedder.input_proj",
                stream,
            )?,
            position_table: tensors
                .get("model.vision_tower.patch_embedder.position_embedding_table")?,
        })
    }

    pub(super) fn forward(
        &self,
        patches: &Array,
        positions: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        validate_positions(patches, positions)?;
        let normalized = patches.multiply_scalar(2.0, stream)?.add_scalar(-1.0, stream)?;
        let projected = self.projection.forward(&normalized, stream)?;
        let x = position_axis(positions, 0, stream)?;
        let y = position_axis(positions, 1, stream)?;
        let x_table = slice_axis(&self.position_table, 0, 0, 1, stream)?.squeeze_axis(0, stream)?;
        let y_table = slice_axis(&self.position_table, 0, 1, 2, stream)?.squeeze_axis(0, stream)?;
        let position = x_table.take(&x, 0, stream)?.add(&y_table.take(&y, 0, stream)?, stream)?;
        projected.add(&position, stream)
    }
}

pub(super) fn position_axis(positions: &Array, axis: usize, stream: &Stream) -> Result<Array> {
    slice_axis(positions, 2, axis, axis + 1, stream)?.squeeze_axis(2, stream)
}

fn validate_positions(patches: &Array, positions: &Array) -> Result<()> {
    let patch_shape = patches.shape()?;
    let position_shape = positions.shape()?;
    if patch_shape.len() == 3
        && position_shape.len() == 3
        && patch_shape[..2] == position_shape[..2]
        && position_shape[2] == 2
    {
        return Ok(());
    }
    Err(Error::InvalidModel(format!(
        "pooled vision patches {patch_shape:?} require position IDs [batch, sequence, 2], got {position_shape:?}"
    )))
}
