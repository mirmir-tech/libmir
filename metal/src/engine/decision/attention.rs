use super::batch::PaddedBatch;
use crate::engine::{Array, Result, RopeOptions, Stream};

/// Multi-head attention over a fused `[rows, length, 3 × hidden]` projection
/// laid out as query, key, value, each head-major.
pub struct Heads {
    pub heads: i32,
    pub head_dim: i32,
}

impl Heads {
    /// Returns `[rows, length, hidden]`; `rope_base` rotates queries and keys
    /// in the rotate-half layout.
    pub fn attend(
        &self,
        qkv: &Array,
        batch: &PaddedBatch,
        rope_base: Option<f32>,
        mask: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.heads * self.head_dim;
        let part = |index: i32| -> Result<Array> {
            let (rows, length) = (usize::try_from(batch.rows)?, usize::try_from(batch.length)?);
            let start = usize::try_from(index * hidden)?;
            let stop = start + usize::try_from(hidden)?;
            let heads = qkv
                .slice(&[0, 0, start], &[rows, length, stop], stream)?
                .reshape(&[batch.rows, batch.length, self.heads, self.head_dim], stream)?
                .transpose(&[0, 2, 1, 3], stream)?;
            match (index, rope_base) {
                (0 | 1, Some(base)) => heads.rope(
                    RopeOptions {
                        dimensions: self.head_dim,
                        traditional: false,
                        base: Some(base),
                        scale: 1.0,
                        offset: 0,
                    },
                    stream,
                ),
                _ => Ok(heads),
            }
        };
        let scale = 1.0 / f32::from(u16::try_from(self.head_dim)?).sqrt();
        part(0)?
            .masked_scaled_dot_product_attention(&part(1)?, &part(2)?, scale, mask, stream)?
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[batch.rows, batch.length, hidden], stream)
    }
}
