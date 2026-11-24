//! The clamped-routed CUDA geometry, from the plan admission accepted.

use architecture::{
    cuda::{CudaDecoderPlan, CudaLayer},
    lowering::{Activation, FeedForwardLowering},
};
use models::semantic::{
    AttentionSpec, DecoderLayerSpec, FeedForwardSpec, MixerSpec, NormalizationKind,
    PositionEncodingSpec, RopeScalingSpec, RotaryLayoutSpec, SemanticModelSpec,
};
use runtime::kv::{CacheConfig, KvStorageSpec};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct ClampedRoutedConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub epsilon: f32,
    pub scale: f32,
    pub theta: f32,
    pub factor: f32,
    pub initial_context: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub rope_truncate: bool,
    pub swiglu_limit: f32,
}

impl ClampedRoutedConfig {
    /// The geometry of `spec`, run as `plan`, the plan admission accepted.
    pub(super) fn from_admitted(plan: &CudaDecoderPlan, spec: &SemanticModelSpec) -> Result<Self> {
        let (Some(first), Some(lowered)) = (spec.decoder.layers.first(), plan.layers().first())
        else {
            return Err(Error::UnsupportedDecoderLayer("empty clamped-routed decoder".into()));
        };
        let (attention, intermediate, experts, top_k, clamp) = layer_contract(first, lowered)?;
        let (theta, factor, initial_context, beta_fast, beta_slow, rope_truncate) =
            rotary(attention)?;
        for (layer, lowered) in spec.decoder.layers.iter().zip(plan.layers()) {
            let contract = layer_contract(layer, lowered)?;
            validate_uniform(contract, attention, intermediate, experts, top_k)?;
            let position = rotary(match &layer.mixer {
                MixerSpec::SoftmaxAttention(attention) => attention,
                MixerSpec::LinearAttention(_)
                | MixerSpec::StateSpace(_)
                | MixerSpec::LatentAttention(_) => unreachable!("validated clamped-routed mixer"),
            })?;
            if position != (theta, factor, initial_context, beta_fast, beta_slow, rope_truncate) {
                return Err(Error::UnsupportedDecoderLayer(format!(
                    "clamped-routed layer {} has non-uniform rotary geometry",
                    layer.index
                )));
            }
        }
        let config = Self {
            vocab: spec.decoder.vocab_size,
            hidden: spec.decoder.hidden_size,
            intermediate,
            query_heads: attention.query_heads,
            kv_heads: attention.key_value_heads,
            head_dim: attention.head_dim,
            experts,
            top_k,
            epsilon: first.input_norm.epsilon.to_string().parse()?,
            scale: attention.scale.to_string().parse()?,
            theta,
            factor,
            initial_context,
            beta_fast,
            beta_slow,
            rope_truncate,
            swiglu_limit: clamp.to_string().parse()?,
        };
        config.validate_semantics(spec)?;
        Ok(config)
    }

    pub(super) fn storage(self, cache: CacheConfig) -> KvStorageSpec {
        KvStorageSpec::new(cache, self.kv_heads, self.head_dim)
    }

    fn validate_semantics(self, spec: &SemanticModelSpec) -> Result<()> {
        let geometry = self.hidden > 0
            && self.intermediate > 0
            && self.query_heads > 0
            && self.kv_heads > 0
            && self.query_heads.is_multiple_of(self.kv_heads)
            && self.head_dim > 0
            && self.experts > 0
            && self.top_k > 0
            && self.top_k <= self.experts;
        let norms = spec.decoder.final_norm.kind == NormalizationKind::Rms
            && approx(spec.decoder.final_norm.epsilon, f64::from(self.epsilon))
            && spec.decoder.layers.iter().all(|layer| {
                layer.input_norm.kind == NormalizationKind::Rms
                    && layer.post_attention_norm.kind == NormalizationKind::Rms
                    && approx(layer.input_norm.epsilon, f64::from(self.epsilon))
                    && approx(layer.post_attention_norm.epsilon, f64::from(self.epsilon))
            });
        if geometry && norms {
            return Ok(());
        }
        Err(Error::UnsupportedDecoderLayer(format!(
            "invalid clamped-routed semantics hidden={}, intermediate={}, query_heads={}, kv_heads={}, head_dim={}, experts={}, top_k={}: expected RMS normalization, non-empty dimensions, GQA divisibility, and top_k <= experts",
            self.hidden,
            self.intermediate,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.experts,
            self.top_k,
        )))
    }
}

/// One layer's geometry from the spec, its experts and clamp from the plan
/// admission accepted; admission already holds the sinks, separate values,
/// plain Q/K, softmax router and clamp slope.
fn layer_contract<'a>(
    layer: &'a DecoderLayerSpec,
    lowered: &CudaLayer,
) -> Result<(&'a AttentionSpec, usize, usize, usize, f64)> {
    let refuse = |what: &str| {
        Err(Error::UnsupportedDecoderLayer(format!(
            "clamped-routed layer {} {what}",
            layer.index
        )))
    };
    let (MixerSpec::SoftmaxAttention(attention), FeedForwardSpec::Routed { routed, .. }) =
        (&layer.mixer, &layer.feed_forward)
    else {
        return refuse("is not softmax attention with routed experts");
    };
    let FeedForwardLowering::Routed { experts, shared: None } = lowered.feed_forward else {
        return refuse("was not admitted as routed experts without a shared expert");
    };
    let Activation::SwiGlu { clamp: Some(clamp) } = experts.activation else {
        return refuse("was not admitted with clamped SwiGLU");
    };
    if !attention.projection_bias {
        return refuse("lacks the attention projection bias");
    }
    Ok((attention, routed.intermediate_size, experts.count, experts.top_k, clamp.limit))
}

fn validate_uniform(
    current: (&AttentionSpec, usize, usize, usize, f64),
    first: &AttentionSpec,
    intermediate: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    let (attention, current_intermediate, current_experts, current_top_k, _) = current;
    if attention.query_heads == first.query_heads
        && attention.key_value_heads == first.key_value_heads
        && attention.head_dim == first.head_dim
        && approx(attention.scale, first.scale)
        && current_intermediate == intermediate
        && current_experts == experts
        && current_top_k == top_k
    {
        return Ok(());
    }
    Err(Error::UnsupportedDecoderLayer(
        "clamped-routed CUDA kernels require uniform layer geometry".into(),
    ))
}

fn rotary(attention: &AttentionSpec) -> Result<(f32, f32, f32, f32, f32, bool)> {
    let PositionEncodingSpec::Rotary(rotary) = &attention.position else {
        return Err(Error::UnsupportedDecoderLayer(
            "clamped-routed CUDA kernels require rotary positions".into(),
        ));
    };
    let Some(RopeScalingSpec::Yarn {
        factor,
        beta_fast,
        beta_slow,
        original_context_len,
        truncate,
        ..
    }) = rotary.scaling
    else {
        return Err(Error::UnsupportedDecoderLayer(
            "clamped-routed CUDA kernels require YaRN scaling".into(),
        ));
    };
    if rotary.layout != RotaryLayoutSpec::Standard || !approx(rotary.partial_factor, 1.0) {
        return Err(Error::UnsupportedDecoderLayer(
            "clamped-routed CUDA kernels require full standard rotary layout".into(),
        ));
    }
    Ok((
        rotary.theta.to_string().parse()?,
        factor.to_string().parse()?,
        original_context_len.to_string().parse()?,
        beta_fast.to_string().parse()?,
        beta_slow.to_string().parse()?,
        truncate,
    ))
}

fn approx(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-6
}

#[cfg(test)]
mod tests;
