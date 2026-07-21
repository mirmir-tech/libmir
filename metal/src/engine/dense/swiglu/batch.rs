use super::{
    attention,
    layer::DenseSwiGluLayer,
    model::{DenseSwiGluModel, OutputProjection},
    weights::AttentionWeights,
};
use crate::engine::{
    Array, DecoderCache, Error, KvCache, Result, Stream, decode_graph,
    decoder::forward_packed_layers, native_paged_attention_mode, paged_attention_min_context,
};

impl DenseSwiGluModel {
    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        let hidden = forward_packed_layers(&self.layers, hidden, caches, positions, stream)?;
        let hidden = self.final_norm.apply(&hidden, 1.0e-6, stream)?;
        let logits = match &self.output_projection {
            OutputProjection::TiedEmbedding => self.embedding.project(&hidden, stream)?,
            OutputProjection::Linear(output) => output.forward(&hidden, stream)?,
        };
        decode_graph::export_once(&logits, stream)?;
        Ok(logits)
    }
}

impl DenseSwiGluLayer {
    pub(super) fn mix_packed(
        &self,
        input: &Array,
        caches: &mut [&mut KvCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention = packed_attention(
            &normalized,
            &self.weights.attention,
            self.fused_attention.as_ref(),
            self.config,
            caches,
            positions,
            stream,
        )?;
        input.add(&attention, stream)
    }

    pub(super) fn feed_forward_packed(
        &self,
        input: &Array,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let normalized =
            self.weights
                .post_attention_norm
                .apply(input, self.config.rms_norm_eps, stream)?;
        let fused = (batch_size <= 2).then_some(self.fused_gate_up.as_ref()).flatten();
        let (gate, up) = fused.map_or_else(
            || {
                Ok((
                    self.weights.mlp.gate.forward(&normalized, stream)?,
                    self.weights.mlp.up.forward(&normalized, stream)?,
                ))
            },
            |fused| fused.forward_pair(&normalized, stream),
        )?;
        let activated = gate.silu_mul(&up, stream)?;
        input.add(&self.weights.mlp.down.forward(&activated, stream)?, stream)
    }
}

#[allow(clippy::too_many_arguments)]
fn packed_attention(
    input: &Array,
    weights: &AttentionWeights,
    fused: Option<&crate::engine::FusedAttention>,
    config: super::config::DenseSwiGluLayerConfig,
    caches: &mut [&mut KvCache],
    positions: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let batch = i32::try_from(caches.len())?;
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
    let queries = queries.reshape(&[batch, 1, config.heads, config.head_dim], stream)?;
    let queries =
        attention::normalize(queries, weights.query_norm.as_ref(), config.rms_norm_eps, stream)?;
    let keys = keys.reshape(&[batch, 1, config.kv_heads, config.head_dim], stream)?;
    let keys = attention::normalize(keys, weights.key_norm.as_ref(), config.rms_norm_eps, stream)?;
    let values = values
        .reshape(&[batch, 1, config.kv_heads, config.head_dim], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    let rows = packed_attention_rows(
        &queries,
        &keys,
        &values,
        caches,
        AttentionRows { weights, config, positions, stream },
    )?;
    let rows = rows.iter().collect::<Vec<_>>();
    let output = Array::concatenate(&rows, 0, stream)?.transpose(&[0, 2, 1, 3], stream)?;
    let width = config.heads * config.head_dim;
    weights.output.forward(&output.reshape(&[batch, 1, width], stream)?, stream)
}

fn packed_attention_rows(
    queries: &Array,
    keys: &Array,
    values: &Array,
    caches: &mut [&mut KvCache],
    context: AttentionRows<'_>,
) -> Result<Vec<Array>> {
    let AttentionRows { weights, config, positions, stream } = context;
    caches
        .iter_mut()
        .enumerate()
        .map(|(row, cache)| {
            let position = positions[row];
            let query = sequence_row_slice(queries, row, config.heads, config.head_dim, stream)?;
            let query = attention::rope_layout(
                &query,
                weights.rope_frequencies.as_ref(),
                config,
                position,
                stream,
            )?;
            let key = sequence_row_slice(keys, row, config.kv_heads, config.head_dim, stream)?;
            let key = attention::rope_layout(
                &key,
                weights.rope_frequencies.as_ref(),
                config,
                position,
                stream,
            )?;
            let value = row_slice(values, row, config.kv_heads, config.head_dim, stream)?;
            let mode = native_paged_attention_mode(
                config.head_dim,
                config.heads,
                config.kv_heads,
                usize::try_from(position)? + 1,
                stream.config().cache.force_native_paged_attention,
            );
            let context = cache.update_for_attention_mode(
                &key,
                &value,
                stream,
                paged_attention_min_context(stream),
                mode,
            )?;
            match context.paged {
                Some(paged) => query.paged_scaled_dot_product_attention_with_scratch(
                    paged.attention(),
                    paged.scratch(),
                    config.attention_scale,
                    stream,
                ),
                None => query.scaled_dot_product_attention(
                    &context.keys,
                    &context.values,
                    config.attention_scale,
                    false,
                    stream,
                ),
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct AttentionRows<'a> {
    weights: &'a AttentionWeights,
    config: super::config::DenseSwiGluLayerConfig,
    positions: &'a [i32],
    stream: &'a Stream,
}

fn row_slice(
    input: &Array,
    row: usize,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(heads)?, 1, usize::try_from(head_dim)?],
        stream,
    )
}

fn sequence_row_slice(
    input: &Array,
    row: usize,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, 1, usize::try_from(heads)?, usize::try_from(head_dim)?],
        stream,
    )
}
