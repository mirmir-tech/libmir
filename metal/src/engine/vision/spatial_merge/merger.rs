use super::dimension;
use crate::engine::{Array, DenseLinear, Error, LayerNorm, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct PatchMerger {
    norm: LayerNorm,
    input: DenseLinear,
    output: DenseLinear,
    hidden_size: usize,
    merge: usize,
}

impl PatchMerger {
    pub(super) fn load(
        tensors: &ModelTensors,
        hidden_size: usize,
        merge: usize,
        prefix: &str,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::load(tensors, &format!("{prefix}.merger.norm"), 1.0e-6)?,
            input: DenseLinear::load(tensors, &format!("{prefix}.merger.linear_fc1"), stream)?,
            output: DenseLinear::load(tensors, &format!("{prefix}.merger.linear_fc2"), stream)?,
            hidden_size,
            merge,
        })
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let shape = input.shape()?;
        let sequence = usize::try_from(*shape.get(1).ok_or(Error::ShapeOverflow)?)?;
        let unit = self.merge * self.merge;
        if !sequence.is_multiple_of(unit) {
            return Err(Error::InvalidModel(
                "spatial-merge vision patch sequence is incompatible with merger".into(),
            ));
        }
        let normalized = self.norm.forward(input, stream)?.reshape(
            &[
                dimension(sequence / unit, "merged sequence")?,
                dimension(self.hidden_size * unit, "merged hidden size")?,
            ],
            stream,
        )?;
        let output = self
            .output
            .forward(&self.input.forward(&normalized, stream)?.gelu(stream)?, stream)?;
        let output_shape = output.shape()?;
        output.reshape(
            &[
                1,
                dimension(sequence / unit, "merged sequence")?,
                *output_shape.get(1).ok_or(Error::ShapeOverflow)?,
            ],
            stream,
        )
    }
}
