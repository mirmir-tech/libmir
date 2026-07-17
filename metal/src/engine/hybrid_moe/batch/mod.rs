mod attention;

use self::attention::packed_attention;
use super::{HybridMoeLayer, feed_forward, model::HybridMoeModel};
use crate::engine::{Array, DecoderCache, KvCache, Result, Stream, decode_graph};

impl HybridMoeModel {
    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        hidden = hidden.multiply_scalar(self.embed_scale, stream)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let mut layer_caches = caches
                .iter_mut()
                .map(|cache| Ok(&mut cache.attention_caches_mut()?[index]))
                .collect::<Result<Vec<_>>>()?;
            hidden = layer.forward_packed(&hidden, &mut layer_caches, positions, stream)?;
        }
        let hidden = hidden.rms_norm(&self.final_norm, 1.0e-6, stream)?;
        let logits = self.embedding.project(&hidden, stream)?;
        let logits = match self.softcap {
            Some(cap) => logits.logit_softcap(cap, stream)?,
            None => logits,
        };
        decode_graph::export_once(&logits, stream)?;
        Ok(logits)
    }
}

impl HybridMoeLayer {
    fn forward_packed(
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
            stream,
        )?;
        let attention =
            self.weights
                .post_attention_norm
                .apply(&attention, self.config.rms_norm_eps, stream)?;
        let hidden = input.add(&attention, stream)?;
        let feed_forward = feed_forward::forward(
            &hidden,
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
        hidden.add(&feed_forward, stream)?.multiply(&self.weights.layer_scalar, stream)
    }
}
