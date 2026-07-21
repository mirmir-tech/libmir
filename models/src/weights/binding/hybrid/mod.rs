use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    LinearAttentionTensorRole, LogicalTensorRole, TensorBinding, WeightBindingPlan,
};
use crate::error::{ModelsError, Result};

pub(super) mod moe;

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
    pub routed_gate: &'a TensorBinding,
    pub routed_up: &'a TensorBinding,
    pub routed_down: &'a TensorBinding,
    pub shared_gate: &'a TensorBinding,
    pub shared_up: &'a TensorBinding,
    pub shared_down: &'a TensorBinding,
    pub shared_output_gate: &'a TensorBinding,
}

impl WeightBindingPlan {
    pub fn hybrid_decoder_layer(&self, index: usize) -> Result<HybridDecoderLayerBindings<'_>> {
        Ok(HybridDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            mixer: mixer(self, index)?,
            post_attention_norm: layer(self, index, LayerTensorRole::PostAttentionNorm)?,
            feed_forward: feed_forward(self, index)?,
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
    fn bindings(self) -> [&'a TensorBinding; 8] {
        [
            self.router,
            self.routed_gate,
            self.routed_up,
            self.routed_down,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
            self.shared_output_gate,
        ]
    }
}

fn mixer(plan: &WeightBindingPlan, index: usize) -> Result<HybridMixerBindings<'_>> {
    let linear = optional_linear(plan, index, LinearAttentionTensorRole::QkvProjection).is_some();
    let softmax = optional_attention(plan, index, AttentionProjectionRole::Query).is_some();
    match (linear, softmax) {
        (true, false) => linear_bindings(plan, index).map(HybridMixerBindings::Linear),
        (false, true) => softmax_bindings(plan, index).map(HybridMixerBindings::Softmax),
        (true, true) => Err(invalid(index, "both linear and softmax mixer bindings are present")),
        (false, false) => Err(invalid(index, "no mixer binding is present")),
    }
}

fn linear_bindings(plan: &WeightBindingPlan, index: usize) -> Result<LinearAttentionBindings<'_>> {
    let get = |tensor| linear(plan, index, tensor);
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

fn softmax_bindings(plan: &WeightBindingPlan, index: usize) -> Result<GatedSoftmaxBindings<'_>> {
    let get = |projection| attention(plan, index, projection);
    Ok(GatedSoftmaxBindings {
        query: get(AttentionProjectionRole::Query)?,
        key: get(AttentionProjectionRole::Key)?,
        value: get(AttentionProjectionRole::Value)?,
        output: get(AttentionProjectionRole::Output)?,
        query_norm: layer(plan, index, LayerTensorRole::QueryNorm)?,
        key_norm: layer(plan, index, LayerTensorRole::KeyNorm)?,
    })
}

fn feed_forward(
    plan: &WeightBindingPlan,
    index: usize,
) -> Result<SharedRoutedFeedForwardBindings<'_>> {
    let expert = |projection| {
        layer(plan, index, LayerTensorRole::ExpertProjection { expert: None, projection })
    };
    let shared =
        |projection| layer(plan, index, LayerTensorRole::SharedExpertProjection { projection });
    Ok(SharedRoutedFeedForwardBindings {
        router: layer(plan, index, LayerTensorRole::Router)?,
        routed_gate: expert(ExpertProjectionRole::Gate)?,
        routed_up: expert(ExpertProjectionRole::Up)?,
        routed_down: expert(ExpertProjectionRole::Down)?,
        shared_gate: shared(FeedForwardProjectionRole::Gate)?,
        shared_up: shared(FeedForwardProjectionRole::Up)?,
        shared_down: shared(FeedForwardProjectionRole::Down)?,
        shared_output_gate: layer(plan, index, LayerTensorRole::SharedExpertOutputGate)?,
    })
}

fn linear(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LinearAttentionTensorRole,
) -> Result<&TensorBinding> {
    layer(plan, index, LayerTensorRole::LinearAttention { tensor })
}

fn optional_linear(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LinearAttentionTensorRole,
) -> Option<&TensorBinding> {
    optional(plan, index, LayerTensorRole::LinearAttention { tensor })
}

fn attention(
    plan: &WeightBindingPlan,
    index: usize,
    projection: AttentionProjectionRole,
) -> Result<&TensorBinding> {
    layer(plan, index, LayerTensorRole::AttentionProjection { projection })
}

fn optional_attention(
    plan: &WeightBindingPlan,
    index: usize,
    projection: AttentionProjectionRole,
) -> Option<&TensorBinding> {
    optional(plan, index, LayerTensorRole::AttentionProjection { projection })
}

fn layer(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    let role = LogicalTensorRole::Layer { index, tensor };
    plan.binding(&role)
        .ok_or_else(|| invalid(index, &format!("unbound role {role:?}")))
}

fn optional(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&LogicalTensorRole::Layer { index, tensor })
}

fn invalid(index: usize, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid hybrid decoder layer {index}: {reason}"))
}

fn sources<'a>(bindings: impl IntoIterator<Item = &'a TensorBinding>) -> Vec<&'a str> {
    bindings.into_iter().flat_map(TensorBinding::physical_sources).collect()
}
