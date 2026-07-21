use super::{
    layer::{Attention, HybridLinearMoeLayer},
    model::HybridLinearMoeModel,
};
use crate::engine::{
    Array, DecoderCache, Result, Stream, decode_graph,
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
        let attention = self.packed_attention(&normalized, caches, positions, stream)?;
        input.add(&attention, stream)
    }

    fn feed_forward_packed(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let normalized = self.post_attention_norm.apply(input, self.rms_norm_eps, stream)?;
        input.add(&self.moe.forward(&normalized, stream)?, stream)
    }

    fn packed_attention(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        if let Attention::Linear(layer) = &self.attention {
            let mut states = caches
                .iter_mut()
                .map(|cache| cache.gated_delta_state(self.index))
                .collect::<Result<Vec<_>>>()?;
            if let Some(output) = layer.forward_packed(input, &mut states, stream)? {
                return Ok(output);
            }
        }
        let rows = self.attention_rows(input, caches, positions, stream)?;
        let rows = rows.iter().collect::<Vec<_>>();
        Array::concatenate(&rows, 0, stream)
    }

    fn attention_rows(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Vec<Array>> {
        let shape = input.shape()?;
        let hidden = usize::try_from(*shape.get(2).ok_or_else(|| {
            crate::engine::Error::InvalidModel(
                "packed hybrid-linear input must have [batch, sequence, hidden] shape".into(),
            )
        })?)?;
        caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| {
                let position = positions[row];
                let input = input.slice(&[row, 0, 0], &[row + 1, 1, hidden], stream)?;
                match &self.attention {
                    Attention::Linear(layer) => {
                        layer.forward(&input, cache.gated_delta_state(self.index)?, stream)
                    },
                    Attention::Full(layer) => layer.forward(
                        &input,
                        cache.full_attention_cache(self.index)?,
                        paged_attention_min_context(stream),
                        position,
                        false,
                        stream,
                    ),
                }
            })
            .collect()
    }
}
