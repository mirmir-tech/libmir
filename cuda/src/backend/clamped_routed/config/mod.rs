//! Semantic capability admission for clamped-routed CUDA execution.

use models::semantic::{
    ActivationSpec, AttentionSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, NormalizationKind,
    PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec, SemanticModelSpec,
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
    pub swiglu_limit: f32,
}

impl ClampedRoutedConfig {
    pub(super) fn from_semantic(spec: &SemanticModelSpec) -> Result<Self> {
        let first =
            spec.decoder.layers.first().ok_or_else(|| {
                Error::UnsupportedDecoderLayer("empty clamped-routed decoder".into())
            })?;
        let (attention, intermediate, experts, top_k, clamp) = layer_contract(first)?;
        let (theta, factor, initial_context, beta_fast, beta_slow) = rotary(attention)?;
        for layer in &spec.decoder.layers {
            validate_uniform(layer_contract(layer)?, attention, intermediate, experts, top_k)?;
            let position = rotary(match &layer.mixer {
                MixerSpec::SoftmaxAttention(attention) => attention,
                MixerSpec::LinearAttention(_) => unreachable!("validated clamped-routed mixer"),
            })?;
            if position != (theta, factor, initial_context, beta_fast, beta_slow) {
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

fn layer_contract(
    layer: &models::semantic::DecoderLayerSpec,
) -> Result<(&AttentionSpec, usize, usize, usize, f64)> {
    let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "clamped-routed layer {} requires softmax attention",
            layer.index
        )));
    };
    let FeedForwardSpec::Routed { routed, shared: None } = &layer.feed_forward else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "clamped-routed layer {} requires routed experts without a shared expert",
            layer.index
        )));
    };
    let ActivationSpec::SwiGlu { alpha, clamp: Some(clamp), up_shift } = routed.activation else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "clamped-routed layer {} requires clamped SwiGLU",
            layer.index
        )));
    };
    if !attention.sinks
        || !attention.projection_bias
        || attention.key_value_relation != KeyValueRelation::Separate
        || attention.qk_normalization != QkNormalization::None
        || !approx(alpha, 1.702)
        || !approx(up_shift, 1.0)
    {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "clamped-routed layer {} lacks the sink, bias, separate-K/V, or clamped-SwiGLU capability contract",
            layer.index
        )));
    }
    Ok((attention, routed.intermediate_size, routed.expert_count, routed.top_k, clamp))
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

fn rotary(attention: &AttentionSpec) -> Result<(f32, f32, f32, f32, f32)> {
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
    ))
}

fn approx(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-6
}

#[cfg(test)]
mod tests;
