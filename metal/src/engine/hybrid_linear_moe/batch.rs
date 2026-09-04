use super::{
    layer::{Attention, HybridLinearMoeLayer},
    model::HybridLinearMoeModel,
};
use crate::engine::{
    Array, DecoderCache, NATIVE_PAGED_ATTENTION_MIN_CONTEXT, PagedContextMode, Result, Stream,
    decode_graph,
    decoder::{LoweredPackedLayer, forward_packed_layers},
    paged_attention_min_context,
};

impl HybridLinearMoeModel {
    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        let hidden = forward_packed_layers(&self.layers, hidden, caches, positions, stream)?;
        let hidden = self.final_norm.apply(&hidden, self.rms_norm_eps, stream)?;
        let logits = self.output.forward(&hidden, stream)?;
        decode_graph::export_once(&logits, stream)?;
        Ok(logits)
    }

    pub(crate) fn forward_packed_prefill_state(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        let shape = token_ids.shape()?;
        let position = positions.iter().copied().min().map_or(Ok(0), usize::try_from)?;
        let evaluation_step = crate::engine::decoder::prefill_evaluation_step(
            caches.len(),
            usize::try_from(shape[1])?,
            position,
            self.layers.len(),
        );
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.mix_packed_prefill(&hidden, caches, positions, stream)?;
            hidden = layer.feed_forward_packed_prefill(&hidden, stream)?;
            if evaluation_step
                .is_some_and(|step| (index + 1) % step == 0 || index + 1 == self.layers.len())
            {
                hidden.async_eval(stream)?;
                stream.synchronize()?;
            }
        }
        self.final_norm.apply(&hidden, self.rms_norm_eps, stream)
    }
}

impl LoweredPackedLayer for HybridLinearMoeLayer {
    fn forward_packed_mixer(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        _index: usize,
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.mix_packed(input, caches, positions, stream)
    }

    fn forward_packed_feed_forward(
        &self,
        input: &Array,
        _batch_size: usize,
        stream: &Stream,
    ) -> Result<Array> {
        self.feed_forward_packed(input, stream)
    }
}

impl HybridLinearMoeLayer {
    fn mix_packed(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.input_norm.apply(input, self.rms_norm_eps, stream)?;
        let attention = self.packed_attention(&normalized, caches, positions, false, stream)?;
        input.add(&attention, stream)
    }

    fn mix_packed_prefill(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.input_norm.apply(input, self.rms_norm_eps, stream)?;
        let attention = self.packed_attention(&normalized, caches, positions, true, stream)?;
        input.add(&attention, stream)
    }

    fn feed_forward_packed(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let normalized = self.post_attention_norm.apply(input, self.rms_norm_eps, stream)?;
        input.add(&self.moe.forward(&normalized, stream)?, stream)
    }

    fn feed_forward_packed_prefill(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let shape = input.shape()?;
        let [batch, sequence, hidden] = shape.as_slice() else {
            return Err(crate::engine::Error::InvalidModel(
                "packed hybrid-linear prefill must have [batch, sequence, hidden] shape".into(),
            ));
        };
        let tokens = batch.checked_mul(*sequence).ok_or(crate::engine::Error::ShapeOverflow)?;
        let normalized = self.post_attention_norm.apply(input, self.rms_norm_eps, stream)?;
        let flattened = normalized.reshape(&[1, tokens, *hidden], stream)?;
        let moe = self.moe.forward(&flattened, stream)?.reshape(&shape, stream)?;
        input.add(&moe, stream)
    }

    fn packed_attention(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        if !causal && let Attention::Linear(layer) = &self.attention {
            let mut states = caches
                .iter_mut()
                .map(|cache| cache.gated_delta_state(self.index))
                .collect::<Result<Vec<_>>>()?;
            if let Some(output) = layer.forward_packed(input, &mut states, stream)? {
                return Ok(output);
            }
        }
        if causal && let Attention::Full(layer) = &self.attention {
            let mut attention_caches = caches
                .iter_mut()
                .map(|cache| cache.full_attention_cache(self.index))
                .collect::<Result<Vec<_>>>()?;
            return layer.forward_packed_prefill(input, &mut attention_caches, positions, stream);
        }
        if !causal
            && let Attention::Full(layer) = &self.attention
            && caches.len() > 1
            && !use_native_paged_batch(caches.len(), 1, positions)
        {
            let mut attention_caches = caches
                .iter_mut()
                .map(|cache| cache.full_attention_cache(self.index))
                .collect::<Result<Vec<_>>>()?;
            return layer.forward_packed_decode(input, &mut attention_caches, positions, stream);
        }
        let rows = self.attention_rows(input, caches, positions, causal, stream)?;
        let rows = rows.iter().collect::<Vec<_>>();
        Array::concatenate(&rows, 0, stream)
    }

    fn attention_rows(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        causal: bool,
        stream: &Stream,
    ) -> Result<Vec<Array>> {
        let shape = input.shape()?;
        let sequence = usize::try_from(*shape.get(1).ok_or_else(|| {
            crate::engine::Error::InvalidModel(
                "packed hybrid-linear input must have [batch, sequence, hidden] shape".into(),
            )
        })?)?;
        let hidden = usize::try_from(*shape.get(2).ok_or_else(|| {
            crate::engine::Error::InvalidModel(
                "packed hybrid-linear input must have [batch, sequence, hidden] shape".into(),
            )
        })?)?;
        let native_paged_batch = use_native_paged_batch(caches.len(), sequence, positions);
        caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| {
                let position = positions[row];
                let input = input.slice(&[row, 0, 0], &[row + 1, sequence, hidden], stream)?;
                match &self.attention {
                    Attention::Linear(layer) => {
                        layer.forward(&input, cache.gated_delta_state(self.index)?, stream)
                    },
                    Attention::Full(layer) if native_paged_batch => layer.forward_with_mode(
                        &input,
                        cache.full_attention_cache(self.index)?,
                        paged_attention_min_context(stream),
                        position,
                        causal,
                        PagedContextMode::Native,
                        stream,
                    ),
                    Attention::Full(layer) => layer.forward(
                        &input,
                        cache.full_attention_cache(self.index)?,
                        paged_attention_min_context(stream),
                        position,
                        causal,
                        stream,
                    ),
                }
            })
            .collect()
    }
}

fn use_native_paged_batch(batch: usize, sequence: usize, positions: &[i32]) -> bool {
    batch > 1
        && sequence == 1
        && positions.iter().copied().all(|position| {
            usize::try_from(position).is_ok_and(|position| {
                position.saturating_add(1) >= NATIVE_PAGED_ATTENTION_MIN_CONTEXT
            })
        })
}

#[cfg(test)]
mod tests {
    use super::use_native_paged_batch;

    #[test]
    fn long_multi_row_decode_uses_native_paged_attention() {
        assert!(use_native_paged_batch(5, 1, &[8_191; 5]));
        assert!(!use_native_paged_batch(1, 1, &[8_191]));
        assert!(!use_native_paged_batch(5, 2, &[8_191; 5]));
        assert!(!use_native_paged_batch(5, 1, &[8_190; 5]));
    }
}
