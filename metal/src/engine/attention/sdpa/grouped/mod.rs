use crate::engine::{Array, Dtype, Result, Stream};

/// MLX 0.32.4 sends single-token head-dim-512 queries without an array mask to
/// its generic vector kernel, which reads K/V about three times slower than a
/// GQA-grouped matmul, softmax and matmul (Gemma 4 full layers: 5.4 versus
/// 1.6 ms per layer at 32k tokens, 0.68 versus 0.16 ms at 4k).
const GROUPED_HEAD_DIM: i32 = 512;

impl Array {
    /// Decode attention for head dim 512 with an optional additive padding mask
    /// broadcast over heads; `None` when the shapes are outside that case.
    pub(super) fn grouped_decode_attention(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: Option<&Self>,
        stream: &Stream,
    ) -> Result<Option<Self>> {
        let query = self.shape()?;
        let key = keys.shape()?;
        let (
            &[batch, heads, 1, GROUPED_HEAD_DIM],
            &[key_batch, kv_heads, tokens, GROUPED_HEAD_DIM],
        ) = (query.as_slice(), key.as_slice())
        else {
            return Ok(None);
        };
        if key_batch != batch || kv_heads <= 0 || heads % kv_heads != 0 || values.shape()? != key {
            return Ok(None);
        }
        if let Some(mask) = mask
            && (!matches!(mask.dtype()?, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
                || !matches!(mask.shape()?.as_slice(), &[rows, 1, 1, slots] if (rows == batch || rows == 1) && slots == tokens))
        {
            return Ok(None);
        }
        let graph = stream.native().graph();
        let grouped = self
            .reshape(&[batch, kv_heads, heads / kv_heads, GROUPED_HEAD_DIM], stream)?
            .multiply_scalar(scale, stream)?;
        let scores = grouped
            .matmul(&keys.transpose(&[0, 1, 3, 2], stream)?, stream)?
            .astype(Dtype::Float32, stream)?;
        let scores = match mask {
            Some(mask) => scores.add(&mask.astype(Dtype::Float32, stream)?, stream)?,
            None => scores,
        };
        let weights = Self::from_native(graph.softmax(scores.native(), -1, false)?)?
            .astype_like(self, stream)?;
        weights
            .matmul(values, stream)?
            .reshape(&[batch, heads, 1, GROUPED_HEAD_DIM], stream)
            .map(Some)
    }
}

#[cfg(test)]
mod tests;
