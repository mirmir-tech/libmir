use super::{dimension, embedding::position_axis, slice_axis};
use crate::engine::{Array, Dtype, Error, Result, Stream};

#[derive(Debug)]
pub(super) struct VisionRope {
    inverse_frequency: Array,
    head_dim: usize,
}

impl VisionRope {
    pub(super) fn new(head_dim: usize, theta: f64) -> Result<Self> {
        if !head_dim.is_multiple_of(4) || !theta.is_finite() || theta <= 0.0 {
            return Err(Error::InvalidModel(format!(
                "pooled vision head_dim {head_dim} must be divisible by four and rope_theta {theta} must be positive"
            )));
        }
        let spatial_dim = head_dim / 2;
        let spatial_dim_f64 = spatial_dim.to_string().parse::<f64>()?;
        let values = (0..spatial_dim)
            .step_by(2)
            .map(|index| {
                theta
                    .powf(-index.to_string().parse::<f64>()? / spatial_dim_f64)
                    .to_string()
                    .parse::<f32>()
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let length = dimension(values.len(), "RoPE frequency count")?;
        Ok(Self {
            inverse_frequency: Array::from_f32(&values, &[length])?,
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
        let x =
            frequencies(&position_axis(&positions, 0, stream)?, &self.inverse_frequency, stream)?;
        let y =
            frequencies(&position_axis(&positions, 1, stream)?, &self.inverse_frequency, stream)?;
        let cos = Array::concatenate(&[&x, &x, &y, &y], -1, stream)?.cos(stream)?;
        let sin = Array::concatenate(&[&x, &x, &y, &y], -1, stream)?.sin(stream)?;
        let cos = cos.expand_dims(&[1], stream)?.astype_like(query, stream)?;
        let sin = sin.expand_dims(&[1], stream)?.astype_like(query, stream)?;
        Ok((
            rotate_coordinates(query, &cos, &sin, self.head_dim, stream)?,
            rotate_coordinates(key, &cos, &sin, self.head_dim, stream)?,
        ))
    }
}

fn frequencies(positions: &Array, inverse: &Array, stream: &Stream) -> Result<Array> {
    positions.expand_dims(&[2], stream)?.multiply(inverse, stream)
}

fn rotate_coordinates(
    input: &Array,
    cos: &Array,
    sin: &Array,
    head_dim: usize,
    stream: &Stream,
) -> Result<Array> {
    let coordinate_dim = head_dim / 2;
    let mut outputs = Vec::with_capacity(2);
    for coordinate in 0..2 {
        let start = coordinate * coordinate_dim;
        let stop = start + coordinate_dim;
        let input_part = slice_axis(input, 3, start, stop, stream)?;
        let cos_part = slice_axis(cos, 3, start, stop, stream)?;
        let sin_part = slice_axis(sin, 3, start, stop, stream)?;
        let rotated = rotate_half(&input_part, coordinate_dim, stream)?;
        outputs.push(
            input_part
                .multiply(&cos_part, stream)?
                .add(&rotated.multiply(&sin_part, stream)?, stream)?,
        );
    }
    Array::concatenate(&[&outputs[0], &outputs[1]], -1, stream)
}

fn rotate_half(input: &Array, width: usize, stream: &Stream) -> Result<Array> {
    let half = width / 2;
    let first = slice_axis(input, 3, 0, half, stream)?;
    let second = slice_axis(input, 3, half, width, stream)?.multiply_scalar(-1.0, stream)?;
    Array::concatenate(&[&second, &first], -1, stream)
}
