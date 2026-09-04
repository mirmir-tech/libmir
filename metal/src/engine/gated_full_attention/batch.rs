use super::{GatedFullAttention, rope};
use crate::engine::{
    Array, Error, KvCache, PagedContextMode, Result, Stream, fused_gate_up::split_last,
    paged_attention_min_context,
};

impl GatedFullAttention {
    pub(crate) fn forward_packed_prefill(
        &self,
        input: &Array,
        caches: &mut [&mut KvCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_packed(input, caches, positions, true, stream)
    }

    pub(crate) fn forward_packed_decode(
        &self,
        input: &Array,
        caches: &mut [&mut KvCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_packed(input, caches, positions, false, stream)
    }

    fn forward_packed(
        &self,
        input: &Array,
        caches: &mut [&mut KvCache],
        positions: &[i32],
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let batch = dimension(&shape, 0)?;
        let sequence = dimension(&shape, 1)?;
        if usize::try_from(batch)? != caches.len() || caches.len() != positions.len() {
            return Err(Error::InvalidModel(
                "packed gated full-attention row counts differ".into(),
            ));
        }
        let heads = self.config.attention_heads;
        let key_value_heads = self.config.key_value_heads;
        let head_dim = self.config.head_dim;
        let query_width = heads.checked_mul(head_dim).ok_or(Error::ShapeOverflow)?;
        let projected = self.query.forward(input, stream)?.reshape(
            &[batch, sequence, heads, head_dim.checked_mul(2).ok_or(Error::ShapeOverflow)?],
            stream,
        )?;
        let (queries, gate) = split_last(&projected, usize::try_from(head_dim)?, stream)?;
        let queries = self.query_norm.apply(&queries, self.config.rms_norm_eps, stream)?;
        let keys = self
            .key
            .forward(input, stream)?
            .reshape(&[batch, sequence, key_value_heads, head_dim], stream)?;
        let keys = self.key_norm.apply(&keys, self.config.rms_norm_eps, stream)?;
        let values = self
            .value
            .forward(input, stream)?
            .reshape(&[batch, sequence, key_value_heads, head_dim], stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        if let Some(position) = common_position(positions) {
            return self.forward_batched_rows(
                &queries, &gate, &keys, &values, caches, position, batch, sequence,
                key_value_heads, head_dim, query_width, causal, stream,
            );
        }
        let rows = caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| {
                let queries = sequence_row(&queries, row, sequence, heads, head_dim, stream)?;
                let queries = rope(&queries, &self.config, positions[row], None, stream)?;
                let keys = sequence_row(&keys, row, sequence, key_value_heads, head_dim, stream)?;
                let keys = rope(&keys, &self.config, positions[row], None, stream)?;
                let values = head_row(&values, row, sequence, key_value_heads, head_dim, stream)?;
                let context = cache.update_for_attention_mode(
                    &keys,
                    &values,
                    stream,
                    paged_attention_min_context(stream),
                    PagedContextMode::View,
                )?;
                let attended = queries.scaled_dot_product_attention(
                    &context.keys,
                    &context.values,
                    self.config.attention_scale,
                    causal,
                    stream,
                )?;
                let attended = attended
                    .transpose(&[0, 2, 1, 3], stream)?
                    .reshape(&[1, sequence, query_width], stream)?;
                let gate = gate.slice(
                    &[row, 0, 0, 0],
                    &[
                        row + 1,
                        usize::try_from(sequence)?,
                        usize::try_from(heads)?,
                        usize::try_from(head_dim)?,
                    ],
                    stream,
                )?;
                let gate = gate.reshape(&[1, sequence, query_width], stream)?;
                gate.sigmoid_mul(&attended, stream)
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = rows.iter().collect::<Vec<_>>();
        self.output.forward(&Array::concatenate(&rows, 0, stream)?, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batched_rows(
        &self,
        queries: &Array,
        gate: &Array,
        keys: &Array,
        values: &Array,
        caches: &mut [&mut KvCache],
        position: i32,
        batch: i32,
        sequence: i32,
        key_value_heads: i32,
        head_dim: i32,
        query_width: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let queries = rope(queries, &self.config, position, None, stream)?;
        let keys = rope(keys, &self.config, position, None, stream)?;
        let contexts = caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| {
                let keys = head_row(&keys, row, sequence, key_value_heads, head_dim, stream)?;
                let values = head_row(values, row, sequence, key_value_heads, head_dim, stream)?;
                cache.update_for_attention_mode(
                    &keys,
                    &values,
                    stream,
                    paged_attention_min_context(stream),
                    PagedContextMode::View,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let context_keys = contexts.iter().map(|context| &context.keys).collect::<Vec<_>>();
        let context_values = contexts.iter().map(|context| &context.values).collect::<Vec<_>>();
        let keys = Array::concatenate(&context_keys, 0, stream)?;
        let values = Array::concatenate(&context_values, 0, stream)?;
        let attended = queries.scaled_dot_product_attention(
            &keys,
            &values,
            self.config.attention_scale,
            causal,
            stream,
        )?;
        let attended = attended
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, sequence, query_width], stream)?;
        let gate = gate.reshape(&[batch, sequence, query_width], stream)?;
        self.output.forward(&gate.sigmoid_mul(&attended, stream)?, stream)
    }
}

pub(super) fn common_position(positions: &[i32]) -> Option<i32> {
    let position = *positions.first()?;
    positions.iter().all(|candidate| *candidate == position).then_some(position)
}

fn sequence_row(
    input: &Array,
    row: usize,
    sequence: i32,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(sequence)?, usize::try_from(heads)?, usize::try_from(head_dim)?],
        stream,
    )
}

fn head_row(
    input: &Array,
    row: usize,
    sequence: i32,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(heads)?, usize::try_from(sequence)?, usize::try_from(head_dim)?],
        stream,
    )
}

fn dimension(shape: &[i32], axis: usize) -> Result<i32> {
    shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::InvalidModel("packed attention input rank is invalid".into()))
}
