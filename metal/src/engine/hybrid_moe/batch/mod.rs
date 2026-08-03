mod attention;

use self::attention::packed_attention;
use super::{HybridMoeLayer, feed_forward, model::HybridMoeModel};
use crate::engine::{
    Array, DecoderCache, Error, KvCache, Result, Stream, decode_graph,
    decoder::{LoweredPackedLayer, forward_packed_layers},
};

impl HybridMoeModel {
    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        let hidden = hidden.multiply_scalar(self.embed_scale, stream)?;
        let hidden = forward_packed_layers(&self.layers, hidden, caches, positions, stream)?;
        let hidden = hidden.rms_norm(&self.final_norm, 1.0e-6, stream)?;
        let logits = self.embedding.project(&hidden, stream)?;
        let logits = match self.softcap {
            Some(cap) => logits.logit_softcap(cap, stream)?,
            None => logits,
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
        hidden = hidden.multiply_scalar(self.embed_scale, stream)?;
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
                        Error::InvalidModel(format!("missing cache for layer {index}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            hidden = layer.mix_packed_prefill(&hidden, &mut layer_caches, positions, stream)?;
            hidden = layer.feed_forward_packed(&hidden, stream)?;
            if evaluation_step
                .is_some_and(|step| (index + 1) % step == 0 || index + 1 == self.layers.len())
            {
                hidden.async_eval()?;
                stream.synchronize()?;
            }
        }
        Ok(hidden)
    }
}

impl LoweredPackedLayer for HybridMoeLayer {
    fn forward_packed_mixer(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        index: usize,
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mut layer_caches = caches
            .iter_mut()
            .map(|cache| {
                cache
                    .attention_caches_mut()?
                    .get_mut(index)
                    .ok_or_else(|| Error::InvalidModel(format!("missing cache for layer {index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        self.mix_packed(input, &mut layer_caches, positions, stream)
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

impl HybridMoeLayer {
    fn mix_packed(
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
            self.config,
            self.fused_attention.as_ref(),
            self.fused_key_value.as_ref(),
            caches,
            positions,
            false,
            stream,
        )?;
        let attention =
            self.weights
                .post_attention_norm
                .apply(&attention, self.config.rms_norm_eps, stream)?;
        input.add(&attention, stream)
    }

    fn mix_packed_prefill(
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
            self.config,
            self.fused_attention.as_ref(),
            self.fused_key_value.as_ref(),
            caches,
            positions,
            true,
            stream,
        )?;
        let attention =
            self.weights
                .post_attention_norm
                .apply(&attention, self.config.rms_norm_eps, stream)?;
        input.add(&attention, stream)
    }

    fn feed_forward_packed(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let feed_forward = feed_forward::forward(
            input,
            &self.weights,
            self.config,
            self.fused_gate_up.as_ref(),
            self.fused_expert_gate_up.as_ref(),
            stream,
        )?;
        let feed_forward = self.weights.post_feed_forward_norm.apply(
            &feed_forward,
            self.config.rms_norm_eps,
            stream,
        )?;
        input.add(&feed_forward, stream)?.multiply(&self.weights.layer_scalar, stream)
    }
}
