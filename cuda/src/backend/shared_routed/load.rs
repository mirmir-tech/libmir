use models::{
    layout::{AttentionLayerType, DecoderConfig},
    weights::{HybridDecoderLayerBindings, HybridMixerBindings, WeightBindingPlan},
};

use super::SharedRoutedLayerTemplate;
use crate::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, AffineGatedDeltaMoeLayerConfig,
    AffineGatedFullAttentionConfig, AffineGatedFullAttentionMoeLayerConfig,
    AffineGatedFullAttentionWeights, AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights,
    CudaAffineGatedDeltaLayer, CudaAffineGatedDeltaMoeLayer, CudaAffineGatedFullAttention,
    CudaAffineGatedFullAttentionMoeLayer, CudaAffineSharedExpertMoe, CudaBackend, CudaTensor,
    CudaTensorDType, CudaTensorSet, Error, GatedActivation, Result,
};

pub(super) fn build_layer(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    layer: usize,
    bindings: HybridDecoderLayerBindings<'_>,
    norm_shift: f32,
) -> Result<SharedRoutedLayerTemplate> {
    let moe_weights = AffineSharedExpertMoeWeights::load_bindings(tensors, bindings.feed_forward)?;
    let moe = moe_config(decoder, &moe_weights)?;
    let epsilon = decoder.rms_norm_eps.to_string().parse()?;
    match decoder.layer_type(layer) {
        AttentionLayerType::Linear => {
            let linear = decoder.linear_attention.as_ref().ok_or_else(|| {
                Error::UnsupportedDecoderLayer("missing parsed linear attention geometry".into())
            })?;
            let HybridMixerBindings::Linear(mixer_bindings) = bindings.mixer else {
                return Err(Error::UnsupportedDecoderLayer(
                    "linear layer has no linear mixer bindings".into(),
                ));
            };
            let weights = AffineGatedDeltaLayerWeights::load_bindings(tensors, mixer_bindings)?;
            let mixed_width = linear
                .key_heads
                .checked_mul(linear.key_head_dim)
                .and_then(|width| width.checked_mul(2))
                .and_then(|width| {
                    linear
                        .value_heads
                        .checked_mul(linear.value_head_dim)
                        .and_then(|value| width.checked_add(value))
                })
                .ok_or(Error::InvalidDecoderKernel("linear attention width overflow"))?;
            let format = weights.qkv.infer_config(1, decoder.hidden_size, mixed_width)?;
            let attention = AffineGatedDeltaLayerConfig::from_linear_attention(
                decoder.hidden_size,
                linear,
                format.group_size,
                format.bits,
                decoder.rms_norm_eps,
                norm_shift,
            )?;
            let config = AffineGatedDeltaMoeLayerConfig {
                attention,
                moe,
                rms_norm_epsilon: epsilon,
                norm_weight_shift: norm_shift,
            };
            CudaAffineGatedDeltaMoeLayer::new(
                backend,
                config,
                CudaAffineGatedDeltaLayer::new(backend, attention, weights)?,
                CudaAffineSharedExpertMoe::new(backend, moe, moe_weights)?,
                required_norm(tensors, &bindings.input_norm.source, decoder.hidden_size)?,
                required_norm(tensors, &bindings.post_attention_norm.source, decoder.hidden_size)?,
            )
            .map(|value| SharedRoutedLayerTemplate::Linear(Box::new(value)))
        },
        AttentionLayerType::Full => full_layer(
            backend, decoder, tensors, layer, bindings, moe_weights, moe, epsilon, norm_shift,
        ),
        AttentionLayerType::Sliding => Err(Error::UnsupportedDecoderLayer(
            "shared-routed linear mixer stack contains sliding attention".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn full_layer(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    layer: usize,
    bindings: HybridDecoderLayerBindings<'_>,
    moe_weights: AffineSharedExpertMoeWeights,
    moe: AffineSharedExpertMoeConfig,
    epsilon: f32,
    norm_shift: f32,
) -> Result<SharedRoutedLayerTemplate> {
    let HybridMixerBindings::Softmax(mixer) = bindings.mixer else {
        return Err(Error::UnsupportedDecoderLayer(
            "full-attention layer has no softmax mixer bindings".into(),
        ));
    };
    let weights = AffineGatedFullAttentionWeights::load_bindings(tensors, mixer)?;
    let query = decoder
        .num_attention_heads
        .checked_mul(decoder.layer_head_dim(layer))
        .and_then(|width| width.checked_mul(2))
        .ok_or(Error::InvalidDecoderKernel("gated query width overflow"))?;
    let format = weights.query.infer_config(1, decoder.hidden_size, query)?;
    let attention = AffineGatedFullAttentionConfig::from_decoder(
        decoder,
        layer,
        format.group_size,
        format.bits,
        norm_shift,
    )?;
    let config = AffineGatedFullAttentionMoeLayerConfig {
        attention,
        moe,
        rms_norm_epsilon: epsilon,
        norm_weight_shift: norm_shift,
    };
    CudaAffineGatedFullAttentionMoeLayer::new(
        backend,
        config,
        CudaAffineGatedFullAttention::new(backend, attention, weights)?,
        CudaAffineSharedExpertMoe::new(backend, moe, moe_weights)?,
        required_norm(tensors, &bindings.input_norm.source, decoder.hidden_size)?,
        required_norm(tensors, &bindings.post_attention_norm.source, decoder.hidden_size)?,
    )
    .map(|value| SharedRoutedLayerTemplate::Full(Box::new(value)))
}

fn moe_config(
    decoder: &DecoderConfig,
    weights: &AffineSharedExpertMoeWeights,
) -> Result<AffineSharedExpertMoeConfig> {
    let expert_count = decoder
        .num_experts
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing parsed expert count".into()))?;
    let top_k = decoder
        .top_k_experts
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing parsed expert top-k".into()))?;
    let routed = decoder.moe_intermediate_size.ok_or_else(|| {
        Error::UnsupportedDecoderLayer("missing parsed routed expert width".into())
    })?;
    let shared = decoder.shared_expert_intermediate_size.ok_or_else(|| {
        Error::UnsupportedDecoderLayer("missing parsed shared expert width".into())
    })?;
    let expert = weights.routed_gate.infer_config(expert_count, decoder.hidden_size, routed)?;
    let router_format = weights.router.infer_config(1, decoder.hidden_size, expert_count)?;
    Ok(AffineSharedExpertMoeConfig {
        hidden_size: decoder.hidden_size,
        routed_intermediate_size: routed,
        shared_intermediate_size: shared,
        expert_count,
        top_k,
        group_size: expert.group_size,
        expert_bits: expert.bits,
        router_bits: router_format.bits,
        activation: GatedActivation::try_from(decoder)?,
    })
}

pub(super) fn infer_norm_shift(
    tensors: &CudaTensorSet,
    decoder: &DecoderConfig,
    bindings: &WeightBindingPlan,
) -> Result<f32> {
    let index = decoder
        .layer_types
        .iter()
        .position(|kind| *kind == AttentionLayerType::Linear)
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing linear attention layer".into()))?;
    let layer = bindings.hybrid_decoder_layer(index)?;
    let HybridMixerBindings::Linear(linear) = layer.mixer else {
        return Err(Error::UnsupportedDecoderLayer(
            "linear layer has no linear mixer binding".into(),
        ));
    };
    let tensor = tensors
        .get(&linear.convolution.source)
        .ok_or_else(|| Error::MissingTensor(linear.convolution.source.clone()))?;
    Ok(if tensor.shape().last() == Some(&1) {
        0.0
    } else {
        1.0
    })
}

pub(super) fn required_norm(
    tensors: &CudaTensorSet,
    name: &str,
    hidden: usize,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))?;
    if tensor.shape() != [hidden] {
        return Err(Error::InvalidQuantizedTensor {
            name: name.into(),
            expected: vec![hidden],
            actual: tensor.shape().to_vec(),
        });
    }
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch { name: name.into(), expected: "BF16" });
    }
    Ok(tensor)
}
