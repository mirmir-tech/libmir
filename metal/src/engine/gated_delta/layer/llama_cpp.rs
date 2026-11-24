//! llama.cpp's GGUF `DeltaNet` layout (`LinearAttentionLayout::LlamaCpp`)
//! brought back to the Hugging Face one the kernels read.
//!
//! The converter stores `-exp(A_log)` and tiles value heads across key heads
//! (`[K0, K1, .., K0, K1, ..]`) when there are more value than key heads. Load
//! takes the log back and reorders every value-head row (the `v` part of the
//! QKV projection and convolution, `z`, `a`, `b`, `A`, `dt_bias`) to grouped
//! order. The output projection's columns sit inside GGML blocks, so its input
//! goes back to tiled order instead ([`TiledValues::tile`]).

use models::gguf::TiledValueHeads;

use super::GatedDeltaLayerConfig;
use crate::engine::{Array, Dtype, Error, Result, Stream, binding::BoundLinear};

/// Value-head geometry when the stored order differs from grouped order.
#[derive(Debug, Clone, Copy)]
pub(super) struct TiledValues {
    heads: TiledValueHeads,
    head_dim: usize,
}

impl TiledValues {
    pub(super) fn new(config: GatedDeltaLayerConfig) -> Result<Option<Self>> {
        let heads = TiledValueHeads::new(
            usize::try_from(config.key_heads)?,
            usize::try_from(config.value_heads)?,
        )?;
        let head_dim = usize::try_from(config.value_head_dim)?;
        Ok(heads.map(|heads| Self { heads, head_dim }))
    }

    /// Stored rows of every grouped value-head row, for `width` rows a head,
    /// after `offset` leading rows kept in place.
    pub(super) fn rows(self, offset: usize, width: usize) -> Result<Array> {
        let rows = self.heads.stored_rows(offset, width)?;
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
            i32::try_from(self.heads.key_heads())?,
            i32::try_from(self.heads.per_key())?,
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

impl TiledValues {
    /// [`Self::tile`] for a compiled graph: grouped `[batch, 1, heads,
    /// head_dim]` values as tiled `[batch, 1, heads * head_dim]` columns.
    pub(super) fn tile_graph(
        self,
        graph: mirtal::Graph<'_>,
        output: &mirtal::Array,
    ) -> mirtal::Result<mirtal::Array> {
        let batch = output.shape()?.dimensions()[0];
        let split = mirtal::Shape::new([
            batch,
            1,
            self.heads.key_heads(),
            self.heads.per_key(),
            self.head_dim,
        ])?;
        let tiled = graph.transpose(&graph.reshape(output, &split)?, &[0, 1, 3, 2, 4])?;
        let width = self.heads.key_heads() * self.heads.per_key() * self.head_dim;
        graph.reshape(&tiled, &mirtal::Shape::new([batch, 1, width])?)
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
