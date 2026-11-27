//! Gemma-style decode over one persistent `[B, H, T, D]` history per global
//! layer; sliding layers keep their bounded ring and per-row attention.

use super::{
    super::{HybridMoeLayerConfig, attention, weights::AttentionWeights},
    attention::AttentionRows,
};
use crate::engine::{Array, KvCache, Result, Stream, persistent_history};

pub(super) fn persistent(
    [queries, keys, values]: [&Array; 3],
    caches: &mut [&mut KvCache],
    context: AttentionRows<'_>,
) -> Result<Option<(Array, persistent_history::Attended)>> {
    if context.causal {
        return Ok(None);
    }
    persistent_history::attend_publishing(
        &persistent_history::Attend {
            queries,
            keys,
            values,
            scale: 1.0,
            sinks: None,
        },
        caches,
        context.positions,
        |input| rotate(input, context),
        context.stream,
    )
}

/// Projects `[B, H, S, D]` attention output back to the residual width.
pub(super) fn project(
    attended: &Array,
    weights: &AttentionWeights,
    config: HybridMoeLayerConfig,
    [batch, sequence]: [i32; 2],
    stream: &Stream,
) -> Result<Array> {
    let width = config.attention_heads * config.head_dim;
    let output = attended.transpose(&[0, 2, 1, 3], stream)?;
    weights
        .output
        .forward(&output.reshape(&[batch, sequence, width], stream)?, stream)
}

/// `[B, S, H, D]` rows rotated to `[B, H, S, D]` at their positions.
pub(super) fn rotate(input: &Array, context: AttentionRows<'_>) -> Result<Array> {
    let AttentionRows { weights, config, positions, stream, .. } = context;
    let frequencies = weights.rope_frequencies.as_ref();
    if let Some(&first) = positions.first()
        && positions.iter().all(|position| *position == first)
    {
        return attention::rope_layout(input, frequencies, config, first, stream);
    }
    let offsets = positions.iter().map(|p| Ok(u32::try_from(*p)?)).collect::<Result<Vec<_>>>()?;
    let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    let graph = stream.native().graph();
    let rotated = match frequencies {
        Some(frequencies) => graph.rope_with_frequencies_batched(
            input.native(),
            frequencies.native(),
            mirtal::FrequencyRopeOptions {
                dimensions: usize::try_from(config.rope_dimensions)?,
                traditional: config.rope_traditional,
                offset: &offsets,
            },
        )?,
        None => graph.rope_batched(
            input.native(),
            mirtal::RopeOptions {
                dimensions: usize::try_from(config.rope_dimensions)?,
                traditional: config.rope_traditional,
                base: Some(config.rope_base),
                scale: config.rope_position_scale,
                offset: &offsets,
            },
        )?,
    };
    Array::from_native(rotated)?.astype_like(&input, stream)
}
