//! Latent attention projections: the query (direct, or through a normalized
//! query latent) and the per-head key and value up-projections, read once
//! as dense matrices for the absorbed products.

use models::{
    semantic::LatentAttentionSpec,
    weights::{LatentQueryBindings, LatentUpBindings},
};

use crate::engine::{Array, Dtype, ModelTensors, NormWeight, Result, Stream, binding::BoundLinear};

#[derive(Debug)]
pub(super) enum LatentQuery {
    Direct(BoundLinear),
    Latent {
        down: BoundLinear,
        norm: NormWeight,
        up: BoundLinear,
    },
}

impl LatentQuery {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: LatentQueryBindings<'_>,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(match bindings {
            LatentQueryBindings::Direct(query) => {
                Self::Direct(BoundLinear::load(tensors, query, stream)?)
            },
            LatentQueryBindings::Latent { down, norm, up } => Self::Latent {
                down: BoundLinear::load(tensors, down, stream)?,
                norm: NormWeight::load_name(tensors, &norm.source)?,
                up: BoundLinear::load(tensors, up, stream)?,
            },
        })
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream, epsilon: f32) -> Result<Array> {
        match self {
            Self::Direct(query) => query.forward(input, stream),
            Self::Latent { down, norm, up } => {
                let latent = norm.apply(&down.forward(input, stream)?, epsilon, stream)?;
                up.forward(&latent, stream)
            },
        }
    }
}

/// `W_UK` as `[heads, nope, rank]` and `W_UVᵀ` as `[heads, rank, value]`.
#[derive(Debug)]
pub(super) struct LatentUp {
    pub(super) key: Array,
    pub(super) value: Array,
}

impl LatentUp {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: LatentUpBindings<'_>,
        spec: &LatentAttentionSpec,
        stream: &Stream,
    ) -> Result<Self> {
        let heads = i32::try_from(spec.query_heads)?;
        let (nope, value) =
            (i32::try_from(spec.nope_head_dim)?, i32::try_from(spec.value_head_dim)?);
        let rank = i32::try_from(spec.key_value_rank)?;
        let dense = |binding| -> Result<Array> {
            BoundLinear::load(tensors, binding, stream)?
                .dense_matrix(stream)?
                .astype(Dtype::Bfloat16, stream)
        };
        let (key, value_up) = match bindings {
            // `[heads · (nope + value), rank]`: each head's key rows, then its
            // value rows.
            LatentUpBindings::Fused(fused) => {
                let fused = dense(fused)?.reshape(&[heads, nope + value, rank], stream)?;
                let key = fused.slice(&[0, 0, 0], &usize3(heads, nope, rank)?, stream)?;
                let start = [0, usize::try_from(nope)?, 0];
                let value_up = fused.slice(&start, &usize3(heads, nope + value, rank)?, stream)?;
                (key, value_up)
            },
            // GGUF `attn_k_b` is `W_UKᵀ` per head, `[heads, rank, nope]`;
            // `attn_v_b` is `W_UV`, `[heads, value, rank]`.
            LatentUpBindings::Split { key, value: value_binding } => (
                dense(key)?
                    .reshape(&[heads, rank, nope], stream)?
                    .transpose(&[0, 2, 1], stream)?,
                dense(value_binding)?.reshape(&[heads, value, rank], stream)?,
            ),
        };
        let key = contiguous(&key, stream)?;
        let value = contiguous(&value_up.transpose(&[0, 2, 1], stream)?, stream)?;
        for array in [&key, &value] {
            array.async_eval(stream)?;
        }
        Ok(Self { key, value })
    }

    /// `q_nope [.., heads, sequence, nope]` absorbed into latent space.
    pub(super) fn absorb_query(&self, query: &Array, stream: &Stream) -> Result<Array> {
        query.astype_like(&self.key, stream)?.matmul(&self.key, stream)
    }

    /// Attended latents `[.., heads, sequence, rank]` as per-head values.
    pub(super) fn expand_values(&self, attended: &Array, stream: &Stream) -> Result<Array> {
        attended.astype_like(&self.value, stream)?.matmul(&self.value, stream)
    }
}

fn usize3(a: i32, b: i32, c: i32) -> Result<[usize; 3]> {
    Ok([usize::try_from(a)?, usize::try_from(b)?, usize::try_from(c)?])
}

fn contiguous(array: &Array, stream: &Stream) -> Result<Array> {
    Array::from_native(stream.native().graph().contiguous(array.native())?)
}
