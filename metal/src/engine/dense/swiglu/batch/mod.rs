mod attention;

use self::attention::packed_attention;
use super::{
    layer::DenseSwiGluLayer,
    model::{DenseSwiGluModel, OutputProjection},
};
use crate::engine::{
    Array, DecoderCache, KvCache, Result, Stream, decode_graph, decoder::forward_packed_layers,
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
            let mut layer_caches = caches
                .iter_mut()
                .map(|cache| {
                    cache.attention_caches_mut()?.get_mut(index).ok_or_else(|| {
                        crate::engine::Error::InvalidModel(format!(
                            "missing cache for layer {index}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            hidden = layer.mix_packed_prefill(&hidden, &mut layer_caches, positions, stream)?;
            hidden = layer.feed_forward_packed(&hidden, caches.len(), stream)?;
            if evaluation_step
                .is_some_and(|step| (index + 1) % step == 0 || index + 1 == self.layers.len())
            {
                hidden.async_eval(stream)?;
                stream.synchronize()?;
            }
        }
        Ok(hidden)
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
            false,
            stream,
        )?;
        input.add(&attention, stream)
    }

    pub(super) fn mix_packed_prefill(
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
            true,
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
