use models::weights::HybridMoeLayerBindings;

use super::{HybridMoeLayerConfig, weights::LayerWeights};
#[cfg(test)]
use super::{
    attention::{self, DecodeContext},
    feed_forward,
};
#[cfg(test)]
use crate::engine::{
    Array, KvCache,
    lowering::{FeedForwardLowering, MixerLowering, NormalizationLowering},
};
use crate::engine::{
    ExpertFusion, FusedAttention, FusedExpertGateUp, FusedGateUp, FusedKeyValue, ModelTensors,
    Result, Stream,
    binding::BoundLinear,
    fusion_planner::{FusionPlanner, ProjectionBiases},
    lowering::LayerLowering,
};

mod forward;
mod lowered;
#[cfg(test)]
mod tests;

pub(super) use forward::emit_profile;

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
    pub fn load_bindings(
        tensors: &ModelTensors,
        bindings: &HybridMoeLayerBindings<'_>,
        lowering: LayerLowering,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let config = config.validate()?;
        let weights = LayerWeights::load_bindings(tensors, bindings, config, stream)?;
        Self::finish_load(weights, lowering, config, stream)
    }

    #[cfg(test)]
    pub fn load(
        tensors: &ModelTensors,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let config = config.validate()?;
        let weights = LayerWeights::load(tensors, config, stream)?;
        let lowering = LayerLowering {
            index: config.layer_index,
            input_norm: NormalizationLowering::Rms,
            post_attention_norm: NormalizationLowering::Rms,
            mixer: MixerLowering::Softmax { sinks: false, window: config.max_context },
            feed_forward: FeedForwardLowering::DenseAndRouted,
        };
        Self::finish_load(weights, lowering, config, stream)
    }

    fn finish_load(
        weights: LayerWeights,
        lowering: LayerLowering,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let fusion = FusionPlanner::new(stream).projections(
            lowering.feed_forward,
            ProjectionBiases::new(
                [weights.attention.query.has_bias(), weights.attention.key.has_bias()],
                weights.attention.value.as_ref().map(BoundLinear::has_bias),
                [weights.dense.gate.has_bias(), weights.dense.up.has_bias()],
            ),
        );
        let fused_attention = fusion
            .attention
            .then(|| {
                weights.attention.query.fuse_attention(
                    &weights.attention.key,
                    weights.attention.value.as_ref(),
                    stream,
                )
            })
            .transpose()?
            .flatten();
        let fused_key_value = (fusion.key_value && fused_attention.is_none())
            .then(|| weights.attention.key.fuse_key_value(weights.attention.value.as_ref(), stream))
            .transpose()?
            .flatten();
        let fused_gate_up = fusion
            .gate_up
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

    pub(super) fn warm_fused_projections(&self, stream: &Stream) -> Result<()> {
        self.fused_attention.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        self.fused_key_value.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        self.fused_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        self.fused_expert_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))
    }

    pub(super) fn enable_expert_gate_up(&mut self, stream: &Stream) -> Result<bool> {
        if self.fused_expert_gate_up.is_some() {
            return Ok(true);
        }
        let Some((gate, up)) = self.weights.experts.separate() else {
            return Ok(false);
        };
        self.fused_expert_gate_up = gate.fuse_expert_gate_up(up, stream)?;
        self.fused_expert_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        Ok(self.fused_expert_gate_up.is_some())
    }

    pub(super) fn fused_expert_gate_up_bytes(&self) -> Result<Option<usize>> {
        let Some((gate, up)) = self.weights.experts.separate() else {
            return Ok(None);
        };
        gate.fused_expert_gate_up_bytes(up)
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

pub(super) fn profile_components(stream: &Stream) -> bool {
    stream.config().diagnostics.profile_components
}
