//! llama.cpp's GGUF `DeltaNet` layout (`LinearAttentionLayout::LlamaCpp`)
//! brought back to the Hugging Face one the kernels read.
//!
//! The converter stores `-exp(A_log)` and tiles value heads across key heads
//! (`[K0, K1, .., K0, K1, ..]`) when there are more value than key heads. Load
//! takes the log back and reorders every value-head row (the `v` part of the
//! QKV projection and convolution, `z`, `a`, `b`, `A`, `dt_bias`) to grouped
//! order. The output projection's columns sit inside GGML blocks, so its input
//! goes back to tiled order instead ([`TiledValues::tile`]).

use super::GatedDeltaLayerConfig;
use crate::engine::{Array, Dtype, Error, Result, Stream, binding::BoundLinear};

/// Value-head geometry when the stored order differs from grouped order.
#[derive(Debug, Clone, Copy)]
pub(super) struct TiledValues {
    key_heads: usize,
    per_key: usize,
    head_dim: usize,
}

impl TiledValues {
    pub(super) fn new(config: GatedDeltaLayerConfig) -> Result<Option<Self>> {
        let key_heads = usize::try_from(config.key_heads)?;
        let value_heads = usize::try_from(config.value_heads)?;
        Ok((value_heads > key_heads).then(|| Self {
            key_heads,
            per_key: value_heads / key_heads,
            head_dim: usize::try_from(config.value_head_dim).unwrap_or(0),
        }))
    }

    /// Stored rows of grouped head `k * per_key + j`, for `width` rows a head,
    /// after `offset` leading rows kept in place.
    pub(super) fn rows(self, offset: usize, width: usize) -> Result<Array> {
        let heads = self.key_heads * self.per_key;
        let mut rows =
            (0..offset).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
        for grouped in 0..heads {
            let stored = (grouped % self.per_key) * self.key_heads + grouped / self.per_key;
            for row in 0..width {
                rows.push(u32::try_from(offset + stored * width + row)?);
            }
        }
        Array::from_u32(&rows, &[i32::try_from(rows.len())?])
    }

    /// `[.., heads * width]` projection rows (`z`, `a`, `b`) in grouped order.
    pub(super) fn projection(
        self,
        linear: &BoundLinear,
        offset: usize,
        width: usize,
        stream: &Stream,
    ) -> Result<BoundLinear> {
        linear.take_rows(&self.rows(offset, width)?, stream)
    }

    /// A per-head vector or `[channels, ..]` tensor with grouped value heads.
    pub(super) fn leading(
        self,
        tensor: &Array,
        offset: usize,
        width: usize,
        stream: &Stream,
    ) -> Result<Array> {
        tensor.take(&self.rows(offset, width)?, 0, stream)
    }

    /// The `[.., heads * head_dim]` attention output, grouped, in the tiled
    /// order the stored output projection's columns expect.
    pub(super) fn tile(self, output: &Array, stream: &Stream) -> Result<Array> {
        let shape = output.shape()?;
        let [leading @ .., width] = shape.as_slice() else {
            return Err(Error::InvalidModel("DeltaNet output has no feature axis".into()));
        };
        let mut split = leading.to_vec();
        split.extend([
            i32::try_from(self.key_heads)?,
            i32::try_from(self.per_key)?,
            i32::try_from(self.head_dim)?,
        ]);
        let rank = i32::try_from(split.len())?;
        let mut axes = (0..rank).collect::<Vec<_>>();
        axes.swap(usize::try_from(rank - 3)?, usize::try_from(rank - 2)?);
        let mut joined = leading.to_vec();
        joined.push(*width);
        output
            .reshape(&split, stream)?
            .transpose(&axes, stream)?
            .reshape(&joined, stream)
    }
}

/// `A_log` from the stored `-exp(A_log)`, computed once at load.
pub(in crate::engine) fn decay_log(stored: &Array, stream: &Stream) -> Result<Array> {
    let dtype = stored.dtype()?;
    let decay = stored.astype(Dtype::Float32, stream)?.to_vec_f32(stream)?;
    if !decay.iter().all(|value| value.is_finite() && *value < 0.0) {
        return Err(Error::InvalidModel("GGUF ssm_a must be negative".into()));
    }
    let logs = decay.iter().map(|value| (-value).ln()).collect::<Vec<_>>();
    let shape = stored.shape()?;
    Array::from_f32(&logs, &shape)?.astype(dtype, stream)
}
