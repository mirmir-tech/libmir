use super::*;

impl GatedFullAttention {
    pub(super) fn forward_offset_rows(
        &self,
        [queries, gate, keys, values]: [&Array; 4],
        caches: &mut [&mut KvCache],
        positions: &[i32],
        mode: PagedContextMode,
        stream: &Stream,
    ) -> Result<Array> {
        let offsets = positions
            .iter()
            .map(|&p| u32::try_from(p))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
        let rotate = |input: &Array| {
            let input = input.transpose(&[0, 2, 1, 3], stream)?;
            Array::from_native(stream.native().graph().rope_batched(
                input.native(),
                mirtal::RopeOptions {
                    dimensions: usize::try_from(self.config.rope_dimensions)?,
                    traditional: false,
                    base: Some(self.config.rope_base),
                    scale: 1.0,
                    offset: &offsets,
                },
            )?)
        };
        let queries = rotate(queries)?;
        let keys = rotate(keys)?;
        let contexts = caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| {
                let keys = head_row(
                    &keys,
                    row,
                    1,
                    self.config.key_value_heads,
                    self.config.head_dim,
                    stream,
                )?;
                let values = head_row(
                    values,
                    row,
                    1,
                    self.config.key_value_heads,
                    self.config.head_dim,
                    stream,
                )?;
                cache.update_for_attention_mode(
                    &keys,
                    &values,
                    stream,
                    paged_attention_min_context(stream),
                    mode,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Offset arrays batch rotation, not K/V storage: ragged contexts can
        // have different lengths and cannot be concatenated for ordinary SDPA.
        let attended = contexts
            .iter()
            .enumerate()
            .map(|(row, context)| {
                let query = head_row(
                    &queries,
                    row,
                    1,
                    self.config.attention_heads,
                    self.config.head_dim,
                    stream,
                )?;
                attention::row(&query, context, self.config.attention_scale, false, stream)
            })
            .collect::<Result<Vec<_>>>()?;
        let attended = Array::concatenate(&attended.iter().collect::<Vec<_>>(), 0, stream)?;
        let shape =
            [i32::try_from(caches.len())?, 1, self.config.attention_heads * self.config.head_dim];
        let attended = attended.transpose(&[0, 2, 1, 3], stream)?.reshape(&shape, stream)?;
        self.project_output(&gate.reshape(&shape, stream)?.sigmoid_mul(&attended, stream)?, stream)
    }
}
