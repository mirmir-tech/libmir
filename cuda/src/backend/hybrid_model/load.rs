use models::layout::{AttentionLayerType, DecoderConfig};

use super::HybridLayerTemplate;
use crate::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, AffineGatedDeltaMoeLayerConfig,
    AffineGatedFullAttentionConfig, AffineGatedFullAttentionMoeLayerConfig,
    AffineGatedFullAttentionWeights, AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights,
    CudaAffineGatedDeltaMoeLayer, CudaAffineGatedFullAttentionMoeLayer, CudaBackend, CudaTensor,
    CudaTensorDType, CudaTensorSet, Error, GatedActivation, Result,
};

pub(super) fn build_layer(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    layer: usize,
    norm_shift: f32,
) -> Result<HybridLayerTemplate> {
    let prefix = format!("language_model.model.layers.{layer}");
    let moe = moe_config(decoder, tensors, &format!("{prefix}.mlp"))?;
    let epsilon = decoder.rms_norm_eps.to_string().parse()?;
    match decoder.layer_type(layer) {
        AttentionLayerType::Linear => {
            let linear = decoder.linear_attention.as_ref().ok_or_else(|| {
                Error::UnsupportedDecoderLayer("missing parsed linear attention geometry".into())
            })?;
            let weights =
                AffineGatedDeltaLayerWeights::load(tensors, &format!("{prefix}.linear_attn"))?;
            let mixed = linear
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
            let format = weights.qkv.infer_config(1, decoder.hidden_size, mixed)?;
            let attention = AffineGatedDeltaLayerConfig::from_linear_attention(
                decoder.hidden_size,
                linear,
                format.group_size,
                format.bits,
                decoder.rms_norm_eps,
                norm_shift,
            )?;
            CudaAffineGatedDeltaMoeLayer::from_tensors(
                backend,
                tensors,
                &prefix,
                AffineGatedDeltaMoeLayerConfig {
                    attention,
                    moe,
                    rms_norm_epsilon: epsilon,
                    norm_weight_shift: norm_shift,
                },
            )
            .map(|value| HybridLayerTemplate::Linear(Box::new(value)))
        },
        AttentionLayerType::Full => {
            full_layer(backend, decoder, tensors, layer, &prefix, moe, epsilon, norm_shift)
        },
        AttentionLayerType::Sliding => Err(Error::UnsupportedDecoderLayer(
            "hybrid linear stack contains sliding attention".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn full_layer(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    layer: usize,
    prefix: &str,
    moe: AffineSharedExpertMoeConfig,
    epsilon: f32,
    norm_shift: f32,
) -> Result<HybridLayerTemplate> {
    let weights = AffineGatedFullAttentionWeights::load(tensors, &format!("{prefix}.self_attn"))?;
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
    CudaAffineGatedFullAttentionMoeLayer::from_tensors(
        backend,
        tensors,
        prefix,
        AffineGatedFullAttentionMoeLayerConfig {
            attention,
            moe,
            rms_norm_epsilon: epsilon,
            norm_weight_shift: norm_shift,
        },
    )
    .map(|value| HybridLayerTemplate::Full(Box::new(value)))
}

fn moe_config(
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    prefix: &str,
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
    let weights = AffineSharedExpertMoeWeights::load(tensors, prefix)?;
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

pub(super) fn infer_norm_shift(tensors: &CudaTensorSet, decoder: &DecoderConfig) -> Result<f32> {
    let index = decoder
        .layer_types
        .iter()
        .position(|kind| *kind == AttentionLayerType::Linear)
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing linear attention layer".into()))?;
    let name = format!("language_model.model.layers.{index}.linear_attn.conv1d.weight");
    let tensor = tensors.get(&name).ok_or_else(|| Error::MissingTensor(name))?;
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
