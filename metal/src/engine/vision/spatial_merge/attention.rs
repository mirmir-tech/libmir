use super::{dimension, rope::VisionRope, slice_axis};
use crate::engine::{Array, DenseLinear, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct VisionAttention {
    qkv: DenseLinear,
    output: DenseLinear,
    rope: VisionRope,
    heads: usize,
    head_dim: usize,
    scale: f32,
}

impl VisionAttention {
    pub(super) fn load(
        tensors: &ModelTensors,
        prefix: &str,
        heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            qkv: DenseLinear::load(tensors, &format!("{prefix}.qkv"), stream)?,
            output: DenseLinear::load(tensors, &format!("{prefix}.proj"), stream)?,
            rope: VisionRope::new(head_dim)?,
            heads,
            head_dim,
            scale: 1.0 / head_dim.to_string().parse::<f32>()?.sqrt(),
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        positions: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let sequence = shape[1];
        let heads = dimension(self.heads, "attention heads")?;
        let head_dim = dimension(self.head_dim, "attention head width")?;
        let qkv = self
            .qkv
            .forward(input, stream)?
            .reshape(&[1, sequence, 3, heads, head_dim], stream)?;
        let query =
            slice_axis(&qkv, 2, 0, 1, stream)?.reshape(&[1, sequence, heads, head_dim], stream)?;
        let key =
            slice_axis(&qkv, 2, 1, 2, stream)?.reshape(&[1, sequence, heads, head_dim], stream)?;
        let value = slice_axis(&qkv, 2, 2, 3, stream)?
            .reshape(&[1, sequence, heads, head_dim], stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        let (query, key) = self.rope.apply(&query, &key, positions, stream)?;
        let query = query.transpose(&[0, 2, 1, 3], stream)?;
        let key = key.transpose(&[0, 2, 1, 3], stream)?;
        let attention =
            query.scaled_dot_product_attention(&key, &value, self.scale, false, stream)?;
        self.output.forward(
            &attention.transpose(&[0, 2, 1, 3], stream)?.reshape(
                &[1, sequence, dimension(self.heads * self.head_dim, "attention width")?],
                stream,
            )?,
            stream,
        )
    }
}
