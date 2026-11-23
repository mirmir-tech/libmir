use super::{
    AttentionProjectionRole, LayerTensorRole, LinearAttentionTensorRole, LogicalTensorRole,
    TensorBinding, WeightBindingPlan,
    slot::LayerSlot,
    view::{RoutedExpertBindings, sources},
};
use crate::error::{ModelsError, Result};

mod feed_forward;
pub(in crate::weights::binding) use feed_forward::feed_forward;
mod latent;
mod mixed;
pub(super) mod moe;
pub use latent::{
    LatentAttentionBindings, LatentDecoderLayerBindings, LatentQueryBindings, LatentUpBindings,
};
pub use mixed::{DenseFeedForwardBindings, MixedDecoderLayerBindings, MixedFeedForwardBindings};

#[derive(Debug, Clone, Copy)]
pub struct HybridDecoderLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub mixer: HybridMixerBindings<'a>,
    pub post_attention_norm: &'a TensorBinding,
    pub feed_forward: SharedRoutedFeedForwardBindings<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum HybridMixerBindings<'a> {
    Linear(LinearAttentionBindings<'a>),
    Softmax(GatedSoftmaxBindings<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionBindings<'a> {
    pub decay_log: &'a TensorBinding,
    pub convolution: &'a TensorBinding,
    pub time_bias: &'a TensorBinding,
    pub qkv: &'a TensorBinding,
    pub gate: &'a TensorBinding,
    pub alpha: &'a TensorBinding,
    pub beta: &'a TensorBinding,
    pub norm: &'a TensorBinding,
    pub output: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct GatedSoftmaxBindings<'a> {
    pub query: &'a TensorBinding,
    pub key: &'a TensorBinding,
    pub value: &'a TensorBinding,
    pub output: &'a TensorBinding,
    pub query_norm: &'a TensorBinding,
    pub key_norm: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct SharedRoutedFeedForwardBindings<'a> {
    pub router: &'a TensorBinding,
    /// Selection-only bias of a sigmoid router (`e_score_correction_bias`).
    pub router_correction_bias: Option<&'a TensorBinding>,
    pub routed: RoutedExpertBindings<'a>,
    pub shared_gate: &'a TensorBinding,
    pub shared_up: &'a TensorBinding,
    pub shared_down: &'a TensorBinding,
    /// Sigmoid gate on the shared experts' output; absent when they add
    /// ungated.
    pub shared_output_gate: Option<&'a TensorBinding>,
}

impl WeightBindingPlan {
    pub fn hybrid_decoder_layer(&self, index: usize) -> Result<HybridDecoderLayerBindings<'_>> {
        let slot = LayerSlot::Decoder(index);
        Ok(HybridDecoderLayerBindings {
            input_norm: layer(self, slot, LayerTensorRole::InputNorm)?,
            mixer: mixer(self, slot)?,
            post_attention_norm: layer(self, slot, LayerTensorRole::PostAttentionNorm)?,
            feed_forward: feed_forward(self, slot)?,
        })
    }
}

impl<'a> HybridDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = vec![self.input_norm, self.post_attention_norm];
        bindings.extend(self.mixer.bindings());
        bindings.extend(self.feed_forward.bindings());
        sources(bindings)
    }
}

impl<'a> HybridMixerBindings<'a> {
    fn bindings(self) -> Vec<&'a TensorBinding> {
        match self {
            Self::Linear(value) => value.bindings().to_vec(),
            Self::Softmax(value) => value.bindings().to_vec(),
        }
    }
}

impl<'a> LinearAttentionBindings<'a> {
    fn bindings(self) -> [&'a TensorBinding; 9] {
        [
            self.decay_log, self.convolution, self.time_bias, self.qkv, self.gate, self.alpha,
            self.beta, self.norm, self.output,
        ]
    }
}

impl<'a> GatedSoftmaxBindings<'a> {
    fn bindings(self) -> [&'a TensorBinding; 6] {
        [self.query, self.key, self.value, self.output, self.query_norm, self.key_norm]
    }
}

impl<'a> SharedRoutedFeedForwardBindings<'a> {
    pub(in crate::weights::binding) fn bindings(self) -> Vec<&'a TensorBinding> {
        let mut bindings = vec![self.router, self.shared_gate, self.shared_up, self.shared_down];
        bindings.extend(self.router_correction_bias);
        bindings.extend(self.shared_output_gate);
        bindings.extend(self.routed.bindings());
        bindings
    }
}

fn mixer(plan: &WeightBindingPlan, slot: LayerSlot) -> Result<HybridMixerBindings<'_>> {
    let linear = optional_linear(plan, slot, LinearAttentionTensorRole::QkvProjection).is_some();
    let softmax = optional_attention(plan, slot, AttentionProjectionRole::Query).is_some();
    match (linear, softmax) {
        (true, false) => linear_bindings(plan, slot).map(HybridMixerBindings::Linear),
        (false, true) => softmax_bindings(plan, slot).map(HybridMixerBindings::Softmax),
        (true, true) => Err(invalid(slot, "both linear and softmax mixer bindings are present")),
        (false, false) => Err(invalid(slot, "no mixer binding is present")),
    }
}

fn linear_bindings(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
) -> Result<LinearAttentionBindings<'_>> {
    let get = |tensor| linear(plan, slot, tensor);
    Ok(LinearAttentionBindings {
        decay_log: get(LinearAttentionTensorRole::DecayLog)?,
        convolution: get(LinearAttentionTensorRole::Convolution)?,
        time_bias: get(LinearAttentionTensorRole::TimeBias)?,
        qkv: get(LinearAttentionTensorRole::QkvProjection)?,
        gate: get(LinearAttentionTensorRole::GateProjection)?,
        alpha: get(LinearAttentionTensorRole::AlphaProjection)?,
        beta: get(LinearAttentionTensorRole::BetaProjection)?,
        norm: get(LinearAttentionTensorRole::Norm)?,
        output: get(LinearAttentionTensorRole::OutputProjection)?,
    })
}

fn softmax_bindings(plan: &WeightBindingPlan, slot: LayerSlot) -> Result<GatedSoftmaxBindings<'_>> {
    let get = |projection| attention(plan, slot, projection);
    Ok(GatedSoftmaxBindings {
        query: get(AttentionProjectionRole::Query)?,
        key: get(AttentionProjectionRole::Key)?,
        value: get(AttentionProjectionRole::Value)?,
        output: get(AttentionProjectionRole::Output)?,
        query_norm: layer(plan, slot, LayerTensorRole::QueryNorm)?,
        key_norm: layer(plan, slot, LayerTensorRole::KeyNorm)?,
    })
}

fn linear(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LinearAttentionTensorRole,
) -> Result<&TensorBinding> {
    layer(plan, slot, LayerTensorRole::LinearAttention { tensor })
}

fn optional_linear(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LinearAttentionTensorRole,
) -> Option<&TensorBinding> {
    optional(plan, slot, LayerTensorRole::LinearAttention { tensor })
}

fn attention(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    projection: AttentionProjectionRole,
) -> Result<&TensorBinding> {
    layer(plan, slot, LayerTensorRole::AttentionProjection { projection })
}

fn optional_attention(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    projection: AttentionProjectionRole,
) -> Option<&TensorBinding> {
    optional(plan, slot, LayerTensorRole::AttentionProjection { projection })
}

fn layer(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    let role = slot.role(tensor);
    plan.binding(&role)
        .ok_or_else(|| invalid(slot, &format!("unbound role {role:?}")))
}

fn optional(
    plan: &WeightBindingPlan,
    slot: LayerSlot,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&slot.role(tensor))
}

fn invalid(slot: LayerSlot, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid hybrid decoder {slot}: {reason}"))
}
