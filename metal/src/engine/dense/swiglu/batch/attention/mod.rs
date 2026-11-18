mod rows;

use self::rows::{AttentionRows, PackedQueryKeys, packed_attention_rows};
use super::{
    super::{attention, config::DenseSwiGluLayerConfig, weights::AttentionWeights},
    rope,
};
use crate::engine::{Array, Error, FusedAttention, KvCache, Result, Stream};

#[allow(clippy::too_many_arguments)]
pub(super) fn packed_attention(
    input: &Array,
    weights: &AttentionWeights,
    fused: Option<&FusedAttention>,
    config: DenseSwiGluLayerConfig,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    causal: bool,
    stream: &Stream,
) -> Result<Array> {
    let batch = i32::try_from(caches.len())?;
    let sequence = *input
        .shape()?
        .get(1)
        .ok_or_else(|| Error::InvalidModel("packed input has no sequence axis".into()))?;
    let fused = (caches.len() <= 2).then_some(fused).flatten();
    let projections = fused.map_or_else(
        || {
            Ok::<_, Error>((
                weights.query.forward(input, stream)?,
                weights.key.forward(input, stream)?,
                weights.value.forward(input, stream)?,
            ))
        },
        |fused| {
            let output = fused.forward(input, stream)?;
            Ok::<_, Error>((
                output.query,
                output.key,
                output
                    .value
                    .ok_or_else(|| Error::InvalidModel("fused attention omitted values".into()))?,
            ))
        },
    )?;
    let (queries, keys, values) = projections;
    let queries = queries.reshape(&[batch, sequence, config.heads, config.head_dim], stream)?;
    let queries =
        attention::normalize(queries, weights.query_norm.as_ref(), config.rms_norm_eps, stream)?;
    let keys = keys.reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?;
    let keys = attention::normalize(keys, weights.key_norm.as_ref(), config.rms_norm_eps, stream)?;
    let values = values
        .reshape(&[batch, sequence, config.kv_heads, config.head_dim], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    let query_keys = if sequence == 1 {
        let offsets = rope::offsets(positions)?;
        let frequencies = weights.rope_frequencies.as_ref();
        PackedQueryKeys::Rotated {
            queries: rope::rotate_rows(&queries, frequencies, config, &offsets, stream)?,
            keys: rope::rotate_rows(&keys, frequencies, config, &offsets, stream)?,
        }
    } else {
        PackedQueryKeys::Unrotated { queries, keys }
    };
    let rows = packed_attention_rows(
        &query_keys,
        &values,
        caches,
        AttentionRows {
            weights,
            config,
            positions,
            causal,
            stream,
        },
    )?;
    let rows = rows.iter().collect::<Vec<_>>();
    let output = Array::concatenate(&rows, 0, stream)?.transpose(&[0, 2, 1, 3], stream)?;
    let width = config.heads * config.head_dim;
    weights
        .output
        .forward(&output.reshape(&[batch, sequence, width], stream)?, stream)
}
