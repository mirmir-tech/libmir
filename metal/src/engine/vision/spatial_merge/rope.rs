use super::{dimension, slice_axis};
use crate::engine::{Array, Dtype, Error, Result, Stream};

#[derive(Debug)]
pub(super) struct VisionRope {
    inverse_frequency: Array,
    head_dim: usize,
}

impl VisionRope {
    pub(super) fn new(head_dim: usize) -> Result<Self> {
        if !head_dim.is_multiple_of(8) {
            return Err(Error::InvalidModel(format!(
                "spatial-merge vision head width {head_dim} must be divisible by eight"
            )));
        }
        let embedding_dim = head_dim / 2;
        let denominator = embedding_dim.to_string().parse::<f64>()?;
        let values = (0..embedding_dim)
            .step_by(2)
            .map(|index| {
                10_000_f64
                    .powf(-index.to_string().parse::<f64>()? / denominator)
                    .to_string()
                    .parse::<f32>()
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            inverse_frequency: Array::from_f32(
                &values,
                &[dimension(values.len(), "RoPE frequencies")?],
            )?,
            head_dim,
        })
    }

    pub(super) fn apply(
        &self,
        query: &Array,
        key: &Array,
        positions: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let positions = positions.astype(Dtype::Float32, stream)?;
        let y = frequencies(&axis(&positions, 0, stream)?, &self.inverse_frequency, stream)?;
        let x = frequencies(&axis(&positions, 1, stream)?, &self.inverse_frequency, stream)?;
        let spatial = Array::concatenate(&[&y, &x], -1, stream)?;
        let angles = Array::concatenate(&[&spatial, &spatial], -1, stream)?;
        let cos = angles.cos(stream)?;
        let sin = angles.sin(stream)?;
        let cos = cos.expand_dims(&[0, 2], stream)?.astype_like(query, stream)?;
        let sin = sin.expand_dims(&[0, 2], stream)?.astype_like(query, stream)?;
        Ok((
            apply_rope(query, &cos, &sin, self.head_dim, stream)?,
            apply_rope(key, &cos, &sin, self.head_dim, stream)?,
        ))
    }
}

fn axis(positions: &Array, axis: usize, stream: &Stream) -> Result<Array> {
    slice_axis(positions, 1, axis, axis + 1, stream)
}

fn frequencies(positions: &Array, inverse: &Array, stream: &Stream) -> Result<Array> {
    positions.multiply(inverse, stream)
}

fn apply_rope(
    input: &Array,
    cos: &Array,
    sin: &Array,
    head_dim: usize,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape()?;
    let width = usize::try_from(*shape.last().ok_or(Error::ShapeOverflow)?)?;
    if width != head_dim {
        return Err(Error::InvalidModel(format!(
            "spatial-merge vision attention width {width} does not match head width {head_dim}"
        )));
    }
    let half = head_dim / 2;
    let first = slice_axis(input, 3, 0, half, stream)?;
    let second = slice_axis(input, 3, half, head_dim, stream)?;
    let rotated =
        Array::concatenate(&[&second.multiply_scalar(-1.0, stream)?, &first], -1, stream)?;
    input.multiply(cos, stream)?.add(&rotated.multiply(sin, stream)?, stream)
}
