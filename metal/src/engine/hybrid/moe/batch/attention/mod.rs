use super::{
    super::{
        HybridMoeLayerConfig,
        attention::{self, KeyValues},
        sharing::SourceAttention,
        weights::AttentionWeights,
    },
    history,
};
use crate::engine::{Array, Error, FusedAttention, FusedKeyValue, KvCache, Result, Stream};

mod rows;

use rows::{attend_rows, concatenate, prepare_rows, row_query};

#[allow(clippy::too_many_arguments)]
pub(super) fn packed_attention(
    input: &Array,
    weights: &AttentionWeights,
    config: HybridMoeLayerConfig,
    fused_attention: Option<&FusedAttention>,
    fused_key_value: Option<&FusedKeyValue>,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    let batch = i32::try_from(positions.len())?;
    let sequence = *input
        .shape()?
        .get(1)
        .ok_or_else(|| Error::InvalidModel("packed input has no sequence axis".into()))?;
    let attention::Projected { queries, key_values } = attention::project(
        input,
        weights,
        config,
        (fused_attention, fused_key_value),
        [batch, sequence],
        stream,
    )?;
    let context = AttentionRows {
        weights,
        config,
        positions,
        causal,
        stream,
    };
    let output = match key_values {
        KeyValues::Shared(attended) => match attended.as_ref() {
            SourceAttention::Persistent(attended) => {
                attended.attend(&history::rotate(&queries, context)?, 1.0, None, stream)?
            },
            SourceAttention::Rows(rows) => {
                let rows = rows
                    .iter()
                    .enumerate()
                    .map(|(row, (shared, mode))| {
                        Ok((row_query(&queries, row, context)?, shared, *mode))
                    })
                    .collect::<Result<Vec<_>>>()?;
                concatenate(&attend_rows(&rows, causal, stream)?, stream)?
            },
        },
        KeyValues::Own { keys, values, publish } => {
            if let Some((output, attended)) =
                history::persistent([&queries, &keys, &values], caches, context)?
            {
                if let Some(publish) = publish {
                    publish.publish(SourceAttention::Persistent(attended))?;
                }
                output
            } else {
                let prepared = prepare_rows(&queries, &keys, &values, caches, context)?;
                if let Some(publish) = publish {
                    let shared = prepared
                        .iter()
                        .map(|(_, context, mode)| Ok((context.share()?, *mode)))
                        .collect::<Result<Vec<_>>>()?;
                    publish.publish(SourceAttention::Rows(shared))?;
                }
                let rows = prepared
                    .iter()
                    .map(|(query, context, mode)| (query.snapshot(), context, *mode))
                    .map(|(query, context, mode)| Ok((query?, context, mode)))
                    .collect::<Result<Vec<_>>>()?;
                concatenate(&attend_rows(&rows, causal, stream)?, stream)?
            }
        },
    };
    history::project(&output, weights, config, [batch, sequence], stream)
}

#[derive(Clone, Copy)]
pub(super) struct AttentionRows<'a> {
    pub(super) weights: &'a AttentionWeights,
    pub(super) config: HybridMoeLayerConfig,
    pub(super) positions: &'a [i32],
    pub(super) causal: bool,
    pub(super) stream: &'a Stream,
}
