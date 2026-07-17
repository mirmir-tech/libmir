use models::layout::{AttentionLayerType, DecoderConfig, RotaryEmbeddingLayout};
use runtime::kv::{CacheConfig, KvStorageSpec};

use crate::{
    DecodeAttentionConfig, DecodeMoeBlockConfig, DenseSwiGluConfig, Error, GatedActivation, Result,
    kernels::QkvNormalization,
};

/// Session-independent policy used to prepare one BF16 dense `SwiGLU` layer.
#[derive(Clone, Copy, Debug)]
pub struct DenseSwiGluLayerLoadConfig {
    pub cache: CacheConfig,
    pub max_sequence_blocks: usize,
    pub qkv_normalization: QkvNormalization,
    pub projection_format: crate::ProjectionFormat,
}

impl DenseSwiGluLayerLoadConfig {
    pub(super) fn block(self, decoder: &DecoderConfig, layer: usize) -> Result<DenseSwiGluConfig> {
        validate_common(decoder, layer, self.max_sequence_blocks)?;
        Ok(DenseSwiGluConfig {
            attention: attention_config(
                decoder,
                layer,
                self.cache,
                self.max_sequence_blocks,
                self.qkv_normalization,
                self.projection_format,
            )?,
            intermediate_size: decoder.intermediate_size,
            activation: GatedActivation::try_from(decoder)?,
        })
    }
}

/// Session-independent policy used to prepare one NVFP4 routed-MoE layer.
#[derive(Clone, Copy, Debug)]
pub struct NvFp4MoeLayerLoadConfig {
    pub cache: CacheConfig,
    pub max_sequence_blocks: usize,
}

impl NvFp4MoeLayerLoadConfig {
    pub(super) fn block(
        self,
        decoder: &DecoderConfig,
        layer: usize,
    ) -> Result<DecodeMoeBlockConfig> {
        validate_common(decoder, layer, self.max_sequence_blocks)?;
        let experts = decoder.num_experts.ok_or_else(|| unsupported("missing num_experts"))?;
        let top_k = decoder.top_k_experts.ok_or_else(|| unsupported("missing top_k_experts"))?;
        let expert_intermediate = decoder
            .moe_intermediate_size
            .ok_or_else(|| unsupported("missing moe_intermediate_size"))?;
        Ok(DecodeMoeBlockConfig {
            attention: attention_config(
                decoder,
                layer,
                self.cache,
                self.max_sequence_blocks,
                QkvNormalization::ALL,
                crate::ProjectionFormat::Bf16,
            )?,
            dense_intermediate: decoder.intermediate_size,
            expert_intermediate,
            experts,
            top_k,
            router_norm_multiplier: attention_scale(decoder.hidden_size)?,
            activation: GatedActivation::try_from(decoder)?,
        })
    }
}

fn attention_config(
    decoder: &DecoderConfig,
    layer: usize,
    cache: CacheConfig,
    max_sequence_blocks: usize,
    qkv_normalization: QkvNormalization,
    projection_format: crate::ProjectionFormat,
) -> Result<DecodeAttentionConfig> {
    let head_dim = decoder.layer_head_dim(layer);
    let rotary_dim = rotary_dim(decoder, layer, head_dim)?;
    let rope_theta = decoder
        .rope_theta_for_layer(layer)
        .ok_or_else(|| unsupported("missing RoPE theta"))?;
    Ok(DecodeAttentionConfig {
        layer,
        hidden_size: decoder.hidden_size,
        query_heads: decoder.num_attention_heads,
        rotary_dim,
        rope_pairing_dim: pairing_dim(decoder, layer, head_dim, rotary_dim)?,
        rope_theta: to_f32("RoPE theta", rope_theta)?,
        rms_norm_epsilon: to_f32("RMS norm epsilon", decoder.rms_norm_eps)?,
        attention_scale: decoder
            .attention_scale
            .map_or_else(|| attention_scale(head_dim), |scale| to_f32("attention scale", scale))?,
        projection_format,
        qkv_normalization,
        sliding_window: decoder.layer_sliding_window(layer),
        max_sequence_blocks,
        cache: KvStorageSpec::new(cache, decoder.layer_key_value_heads(layer), head_dim),
    })
}

fn validate_common(
    decoder: &DecoderConfig,
    layer: usize,
    max_sequence_blocks: usize,
) -> Result<()> {
    if layer >= decoder.num_hidden_layers {
        return Err(unsupported("layer index exceeds checkpoint depth"));
    }
    if decoder.layer_type(layer) == AttentionLayerType::Linear {
        return Err(unsupported("linear attention is not implemented by this loader"));
    }
    if decoder.rope_layout != RotaryEmbeddingLayout::Standard {
        return Err(unsupported("multi-section RoPE is not implemented by this loader"));
    }
    if !matches!(decoder.rope_type_for_layer(layer), None | Some("default" | "proportional")) {
        return Err(unsupported("checkpoint RoPE type is not implemented by this loader"));
    }
    if max_sequence_blocks == 0 {
        return Err(unsupported("max_sequence_blocks must be greater than zero"));
    }
    Ok(())
}

fn rotary_dim(decoder: &DecoderConfig, layer: usize, head_dim: usize) -> Result<usize> {
    let factor = decoder.partial_rotary_factor_for_layer(layer).unwrap_or(1.0);
    let head = to_f64(head_dim)?;
    let dimensions = head * factor;
    if !dimensions.is_finite() || dimensions <= 0.0 || dimensions.fract() != 0.0 {
        return Err(unsupported("partial RoPE does not produce an integral dimension"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let dimensions = dimensions as usize;
    Ok(dimensions)
}

fn pairing_dim(
    decoder: &DecoderConfig,
    layer: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<usize> {
    match decoder.rope_type_for_layer(layer) {
        Some("proportional") => Ok(head_dim),
        None | Some("default") => Ok(rotary_dim),
        Some(kind) => Err(unsupported(format!("unsupported RoPE type {kind}"))),
    }
}

fn attention_scale(dimension: usize) -> Result<f32> {
    Ok(to_f32("attention dimension", to_f64(dimension)?)?.sqrt().recip())
}

fn to_f64(value: usize) -> Result<f64> {
    let value = u32::try_from(value)?;
    Ok(f64::from(value))
}

fn to_f32(label: &str, value: f64) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(unsupported(format!("{label} is outside finite f32 range")));
    }
    #[allow(clippy::cast_possible_truncation)]
    let value = value as f32;
    Ok(value)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::UnsupportedDecoderLayer(message.into())
}
