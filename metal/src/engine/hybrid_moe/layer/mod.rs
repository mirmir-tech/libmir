use std::time::Instant;

use super::{
    HybridMoeLayerConfig,
    attention::{self, DecodeContext},
    feed_forward,
    weights::LayerWeights,
};
use crate::engine::{
    Array, ExpertFusion, FusedAttention, FusedExpertGateUp, FusedGateUp, FusedKeyValue, KvCache,
    ModelTensors, Result, Stream,
};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct HybridMoeLayer {
    pub(super) config: HybridMoeLayerConfig,
    pub(super) weights: LayerWeights,
    pub(super) fused_attention: Option<FusedAttention>,
    pub(super) fused_key_value: Option<FusedKeyValue>,
    pub(super) fused_gate_up: Option<FusedGateUp>,
    pub(super) fused_expert_gate_up: Option<FusedExpertGateUp>,
}

impl HybridMoeLayer {
    pub fn load(
        tensors: &ModelTensors,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let config = config.validate()?;
        let weights = LayerWeights::load(tensors, config, stream)?;
        let fused_attention = fused_attention_enabled(stream)
            .then(|| {
                weights.attention.query.fuse_attention(
                    &weights.attention.key,
                    weights.attention.value.as_ref(),
                    stream,
                )
            })
            .transpose()?
            .flatten();
        let fused_key_value = (fused_attention_enabled(stream) && fused_attention.is_none())
            .then(|| weights.attention.key.fuse_key_value(weights.attention.value.as_ref(), stream))
            .transpose()?
            .flatten();
        let fused_gate_up = fused_gate_up_enabled(stream)
            .then(|| weights.dense.gate.fuse_gate_up(&weights.dense.up, stream))
            .transpose()?
            .flatten();
        Ok(Self {
            config,
            weights,
            fused_attention,
            fused_key_value,
            fused_gate_up,
            fused_expert_gate_up: None,
        })
    }

    pub fn forward_uncached_decode(&self, input: &Array, stream: &Stream) -> Result<Array> {
        self.forward_decode(input, None, 0, false, stream)
    }

    pub fn forward_decode(
        &self,
        input: &Array,
        cache: Option<&mut KvCache>,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_with_mask(input, cache, position, causal, None, stream)
    }

    pub(super) fn forward_with_mask(
        &self,
        input: &Array,
        cache: Option<&mut KvCache>,
        position: i32,
        causal: bool,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = !causal && profile_components(stream);
        let attention_started = Instant::now();
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention = attention::forward_decode(
            &normalized,
            &self.weights.attention,
            self.config,
            self.fused_attention.as_ref(),
            self.fused_key_value.as_ref(),
            DecodeContext { cache, position, causal, mask, stream },
        )?;
        let attention =
            self.weights
                .post_attention_norm
                .apply(&attention, self.config.rms_norm_eps, stream)?;
        let hidden = input.add(&attention, stream)?;
        if profile {
            emit_profile(&hidden, stream, self.config.layer_index, "attention")?;
            tracing::debug!(
                layer = self.config.layer_index,
                component = "attention",
                milliseconds = attention_started.elapsed().as_secs_f64() * 1_000.0,
                "MLX hybrid MoE component profile"
            );
        }

        let feed_forward_started = Instant::now();
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
        let output = hidden.add(&feed_forward, stream)?;
        let output = output.multiply(&self.weights.layer_scalar, stream)?;
        if profile {
            emit_profile(&output, stream, self.config.layer_index, "feed_forward")?;
            tracing::debug!(
                layer = self.config.layer_index,
                component = "feed_forward",
                milliseconds = feed_forward_started.elapsed().as_secs_f64() * 1_000.0,
                "MLX hybrid MoE component profile"
            );
        }
        Ok(output)
    }

    pub(super) fn warm_fused_projections(&self) -> Result<()> {
        self.fused_attention.as_ref().map_or(Ok(()), FusedAttention::warm)?;
        self.fused_key_value.as_ref().map_or(Ok(()), FusedKeyValue::warm)?;
        self.fused_gate_up.as_ref().map_or(Ok(()), FusedGateUp::warm)?;
        self.fused_expert_gate_up.as_ref().map_or(Ok(()), FusedExpertGateUp::warm)
    }

    pub(super) fn enable_expert_gate_up(&mut self, stream: &Stream) -> Result<bool> {
        if self.fused_expert_gate_up.is_some() {
            return Ok(true);
        }
        self.fused_expert_gate_up = self
            .weights
            .experts
            .gate
            .fuse_expert_gate_up(&self.weights.experts.up, stream)?;
        self.fused_expert_gate_up.as_ref().map_or(Ok(()), FusedExpertGateUp::warm)?;
        Ok(self.fused_expert_gate_up.is_some())
    }

    pub(super) fn fused_expert_gate_up_bytes(&self) -> Result<Option<usize>> {
        self.weights.experts.gate.fused_expert_gate_up_bytes(&self.weights.experts.up)
    }

    #[must_use]
    pub(super) fn fusion_summary(&self) -> (bool, bool, bool, bool) {
        (
            self.fused_attention.is_some(),
            self.fused_key_value.is_some(),
            self.fused_gate_up.is_some(),
            self.fused_expert_gate_up.is_some(),
        )
    }
}

impl ExpertFusion for HybridMoeLayer {
    fn enable_expert_fusion(&mut self, stream: &Stream) -> Result<bool> {
        self.enable_expert_gate_up(stream)
    }

    fn expert_fusion_bytes(&self) -> Result<Option<usize>> {
        self.fused_expert_gate_up_bytes()
    }
}

pub(super) fn fused_gate_up_enabled(stream: &Stream) -> bool {
    stream.config().fusion.hybrid_dense_gate_up.enabled()
}

pub(super) fn fused_attention_enabled(stream: &Stream) -> bool {
    stream.config().fusion.hybrid_attention.enabled()
}

pub(super) fn profile_components(stream: &Stream) -> bool {
    stream.config().diagnostics.profile_components
}

fn emit_profile(output: &Array, stream: &Stream, layer: usize, component: &str) -> Result<()> {
    output.async_eval()?;
    stream.synchronize()?;
    tracing::debug!(layer, component, "MLX hybrid MoE component synchronized");
    Ok(())
}
