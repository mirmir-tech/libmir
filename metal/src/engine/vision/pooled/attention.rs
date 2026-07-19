use super::{dimension, rope::VisionRope};
use crate::engine::{Array, DenseLinear, ModelTensors, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct VisionAttention {
    query: DenseLinear,
    key: DenseLinear,
    value: DenseLinear,
    output: DenseLinear,
    query_norm: NormWeight,
    key_norm: NormWeight,
    rope: VisionRope,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    eps: f32,
}

impl VisionAttention {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        tensors: &ModelTensors,
        prefix: &str,
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            query: DenseLinear::load_clippable(tensors, &format!("{prefix}.q_proj"), stream)?,
            key: DenseLinear::load_clippable(tensors, &format!("{prefix}.k_proj"), stream)?,
            value: DenseLinear::load_clippable(tensors, &format!("{prefix}.v_proj"), stream)?,
            output: DenseLinear::load_clippable(tensors, &format!("{prefix}.o_proj"), stream)?,
            query_norm: NormWeight::load(tensors, &format!("{prefix}.q_norm"))?,
            key_norm: NormWeight::load(tensors, &format!("{prefix}.k_norm"))?,
            rope: VisionRope::new(head_dim, rope_theta)?,
            query_heads,
            key_value_heads,
            head_dim,
            eps,
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        positions: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let batch = shape[0];
        let sequence = shape[1];
        let query_heads = dimension(self.query_heads, "query head count")?;
        let key_value_heads = dimension(self.key_value_heads, "key/value head count")?;
        let head_dim = dimension(self.head_dim, "attention head width")?;

        let query = self
            .query
            .forward(input, stream)?
            .reshape(&[batch, sequence, query_heads, head_dim], stream)?;
        let query = self.query_norm.apply(&query, self.eps, stream)?;
        let key = self
            .key
            .forward(input, stream)?
            .reshape(&[batch, sequence, key_value_heads, head_dim], stream)?;
        let key = self.key_norm.apply(&key, self.eps, stream)?;
        let value = self
            .value
            .forward(input, stream)?
            .reshape(&[batch, sequence, key_value_heads, head_dim], stream)?
            .rms_norm_unit(self.eps, stream)?;

        let query = query.transpose(&[0, 2, 1, 3], stream)?;
        let key = key.transpose(&[0, 2, 1, 3], stream)?;
        let value = value.transpose(&[0, 2, 1, 3], stream)?;
        let (query, key) = self.rope.apply(&query, &key, positions, stream)?;
        let attention = query.scaled_dot_product_attention(&key, &value, 1.0, false, stream)?;
        let hidden = dimension(self.query_heads * self.head_dim, "attention output width")?;
        self.output.forward(
            &attention
                .transpose(&[0, 2, 1, 3], stream)?
                .reshape(&[batch, sequence, hidden], stream)?,
            stream,
        )
    }
}
