//! Decode attention over one persistent `[B, H, T, D]` history per layer.
//! Common-position rows rotate together, append in place and share one SDPA
//! call, so attention cost no longer grows with a dispatch per row.

use super::super::{
    attention as dense_attention, config::DenseSwiGluLayerConfig, weights::AttentionWeights,
};
use crate::engine::{Array, KvCache, Result, Stream, persistent_history};

/// Rotates `[B, S, H, D]` rows into `[B, H, S, D]`, each at its own position.
fn rotate(
    input: &Array,
    config: DenseSwiGluLayerConfig,
    positions: &[i32],
    stream: &Stream,
) -> Result<Array> {
    if let Some(position) =
        positions.first().copied().filter(|first| positions.iter().all(|p| p == first))
    {
        return dense_attention::rope_layout(input, None, config, position, stream);
    }
    let offsets = positions.iter().map(|p| Ok(u32::try_from(*p)?)).collect::<Result<Vec<_>>>()?;
    let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    Array::from_native(stream.native().graph().rope_batched(
        input.native(),
        mirtal::RopeOptions {
            dimensions: usize::try_from(config.head_dim)?,
            traditional: false,
            base: Some(config.rope_base),
            scale: 1.0,
            offset: &offsets,
        },
    )?)
}

/// Returns `None` before touching any cache when per-row attention must run.
#[allow(clippy::too_many_arguments)]
pub(super) fn persistent_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    caches: &mut [&mut KvCache],
    weights: &AttentionWeights,
    config: DenseSwiGluLayerConfig,
    positions: &[i32],
    causal: bool,
    stream: &Stream,
) -> Result<Option<Array>> {
    if causal
        || weights.rope_frequencies.is_some()
        || config.rope_attention_factor.to_bits() != 1.0_f32.to_bits()
    {
        return Ok(None);
    }
    persistent_history::attend(
        &persistent_history::Attend {
            queries,
            keys,
            values,
            scale: config.attention_scale,
            sinks: None,
        },
        caches,
        positions,
        |input| rotate(input, config, positions, stream),
        stream,
    )
}
