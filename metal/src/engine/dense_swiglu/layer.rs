use super::{attention, config::DenseSwiGluLayerConfig, weights::DenseSwiGluWeights};
use crate::engine::{Array, FusedAttention, FusedGateUp, KvCache, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct DenseSwiGluLayer {
    pub(super) config: DenseSwiGluLayerConfig,
    pub(super) weights: DenseSwiGluWeights,
    pub(super) fused_attention: Option<FusedAttention>,
    pub(super) fused_gate_up: Option<FusedGateUp>,
}

impl DenseSwiGluLayer {
    pub(super) fn load(
        tensors: &ModelTensors,
        config: DenseSwiGluLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let weights = DenseSwiGluWeights::load(tensors, config, stream)?;
        let compatible = stream.config().fusion.dense_attention.enabled()
            && !weights.attention.query.has_bias()
            && !weights.attention.key.has_bias()
            && !weights.attention.value.has_bias();
        let fused_attention = compatible
            .then(|| {
                weights.attention.query.fuse_attention(
                    &weights.attention.key,
                    Some(&weights.attention.value),
                    stream,
                )
            })
            .transpose()?
            .flatten();
        let compatible = stream.config().fusion.dense_gate_up.enabled()
            && !weights.mlp.gate.has_bias()
            && !weights.mlp.up.has_bias();
        let fused_gate_up = compatible
            .then(|| weights.mlp.gate.fuse_gate_up(&weights.mlp.up, stream))
            .transpose()?
            .flatten();
        fused_attention.as_ref().map_or(Ok(()), FusedAttention::warm)?;
        fused_gate_up.as_ref().map_or(Ok(()), FusedGateUp::warm)?;
        Ok(Self {
            config,
            weights,
            fused_attention,
            fused_gate_up,
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention = attention::forward(
            &normalized,
            &self.weights.attention,
            self.fused_attention.as_ref(),
            self.config,
            attention::AttentionContext { cache, position, causal, stream },
        )?;
        let hidden = input.add(&attention, stream)?;
        let normalized =
            self.weights
                .post_attention_norm
                .apply(&hidden, self.config.rms_norm_eps, stream)?;
        let fused = (input.shape()?.get(1) == Some(&1))
            .then_some(self.fused_gate_up.as_ref())
            .flatten();
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
        let feed_forward = self.weights.mlp.down.forward(&activated, stream)?;
        hidden.add(&feed_forward, stream)
    }

    pub(super) fn fusion_summary(&self) -> (bool, bool) {
        (self.fused_attention.is_some(), self.fused_gate_up.is_some())
    }
}
