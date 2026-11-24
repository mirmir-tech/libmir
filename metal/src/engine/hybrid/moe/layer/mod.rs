use models::{
    semantic::DecoderLayerSpec,
    weights::{DenseDecoderLayerBindings, HybridMoeLayerBindings},
};

use super::{
    HybridMoeLayerConfig,
    sharing::PerLayerGate,
    weights::{LayerWeights, Publish},
};
#[cfg(test)]
use super::{
    attention::{self, DecodeContext},
    feed_forward,
};
#[cfg(test)]
use crate::engine::KvCache;
use crate::engine::{
    Array, ExpertFusion, FusedAttention, FusedExpertGateUp, FusedGateUp, FusedKeyValue,
    ModelTensors, Result, Stream,
    binding::BoundLinear,
    fusion_planner::{FusionPlanner, ProjectionBiases},
};

mod forward;
mod lowered;
mod shared;
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
    /// Gates the layer's per-layer input into its output (Gemma 4's E
    /// models).
    pub(super) per_layer_gate: Option<PerLayerGate>,
}

impl HybridMoeLayer {
    pub fn load_bindings(
        tensors: &ModelTensors,
        bindings: &HybridMoeLayerBindings<'_>,
        layer: &DecoderLayerSpec,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let config = config.validate()?;
        let frequencies = Array::rope_periods(super::attention(layer)?)?;
        let weights = LayerWeights::load_bindings(tensors, bindings, frequencies, config, stream)?;
        Self::finish_load(weights, config, stream)
    }

    pub(in crate::engine::hybrid::moe) fn load_dense_bindings(
        tensors: &ModelTensors,
        bindings: &DenseDecoderLayerBindings<'_>,
        layer: &DecoderLayerSpec,
        publish: Option<Publish>,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let config = config.validate()?;
        let frequencies = Array::rope_periods(super::attention(layer)?)?;
        let weights = LayerWeights::load_dense_bindings(
            tensors, bindings, frequencies, publish, config, stream,
        )?;
        Self::finish_load(weights, config, stream)
    }

    #[cfg(test)]
    pub fn load(
        tensors: &ModelTensors,
        layer: &super::CheckpointLayer,
        stream: &Stream,
    ) -> Result<Self> {
        let config = layer.config.validate()?;
        let frequencies = Array::rope_periods(super::attention(&layer.spec)?)?;
        let weights = LayerWeights::load(tensors, frequencies, config, stream)?;
        Self::finish_load(weights, config, stream)
    }

    fn finish_load(
        weights: LayerWeights,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let key_values = weights.attention.key_values.projections();
        let fusion = FusionPlanner::new(stream).hybrid_projections(ProjectionBiases::new(
            [weights.attention.query.has_bias(), key_values.is_some_and(|(key, _)| key.has_bias())],
            key_values.and_then(|(_, value)| value).map(BoundLinear::has_bias),
            [weights.dense.gate.has_bias(), weights.dense.up.has_bias()],
        ));
        // A layer reading another's keys and values projects queries alone.
        let fused_attention = key_values
            .filter(|_| fusion.attention)
            .map(|(key, value)| weights.attention.query.fuse_attention(key, value, stream))
            .transpose()?
            .flatten();
        let fused_key_value = key_values
            .filter(|_| fusion.key_value && fused_attention.is_none())
            .map(|(key, value)| key.fuse_key_value(value, stream))
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
            per_layer_gate: None,
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
        let Some((gate, up)) =
            self.weights.routed.as_ref().and_then(|routed| routed.experts.separate())
        else {
            return Ok(false);
        };
        self.fused_expert_gate_up = gate.fuse_expert_gate_up(up, stream)?;
        self.fused_expert_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        Ok(self.fused_expert_gate_up.is_some())
    }

    pub(super) fn fused_expert_gate_up_bytes(&self) -> Result<Option<usize>> {
        let Some((gate, up)) =
            self.weights.routed.as_ref().and_then(|routed| routed.experts.separate())
        else {
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
