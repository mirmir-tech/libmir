use models::{
    layout::{AttentionLayerType, DecoderConfig, RotaryEmbeddingLayout},
    semantic::{AttentionSpec, FeedForwardSpec, MixerSpec, SemanticModelSpec},
};
use runtime::kv::{CacheConfig, KvStorageSpec};

mod routed;

use architecture::{cuda::CudaDecoderPlan, lowering::FeedForwardLowering};
use routed::{dense_router, experts, feed_forward, lowered};

use crate::{
    DecodeAttentionConfig, DecodeMoeBlockConfig, DenseSwiGluConfig, Error, GatedActivation, Result,
};

/// Session-independent policy used to prepare one BF16 dense `SwiGLU` layer.
#[derive(Clone, Copy, Debug)]
pub struct DenseSwiGluLayerLoadConfig {
    pub cache: CacheConfig,
    pub max_sequence_blocks: usize,
    /// Storage of the feed-forward projections, and of the attention
    /// projections unless the checkpoint keeps those in BF16.
    pub projection_format: crate::ProjectionFormat,
}

impl DenseSwiGluLayerLoadConfig {
    pub(super) fn block(
        self,
        decoder: &DecoderConfig,
        (semantic, plan): (&SemanticModelSpec, &CudaDecoderPlan),
        layer: usize,
    ) -> Result<DenseSwiGluConfig> {
        validate_common(decoder, layer, self.max_sequence_blocks)?;
        // A dense layer's own width (Llama 4's `intermediate_size_mlp` beside
        // its experts) or its routed experts', as the semantic spec resolves
        // them; the activation and router as admission lowered them.
        let (intermediate_size, activation, experts) =
            match (feed_forward(semantic, layer)?, lowered(plan, layer)?.feed_forward) {
                (
                    FeedForwardSpec::Dense { intermediate_size, .. },
                    FeedForwardLowering::Dense { activation },
                ) => (*intermediate_size, activation, None),
                (
                    FeedForwardSpec::Routed { routed, shared },
                    FeedForwardLowering::Routed { experts, .. },
                ) => {
                    (routed.intermediate_size, experts.activation, Some((experts, shared.as_ref())))
                },
                _ => return Err(unsupported("dense CUDA layers are gated MLPs or routed experts")),
            };
        Ok(DenseSwiGluConfig {
            attention: attention_config(
                decoder,
                semantic,
                layer,
                self.cache,
                self.max_sequence_blocks,
                self.projection_format,
            )?,
            intermediate_size,
            activation: GatedActivation::try_from(activation)?,
            feed_forward_format: self.projection_format,
            experts: experts
                .map(|(lowered, shared)| {
                    let experts = crate::RoutedExpertsConfig {
                        count: lowered.count,
                        top_k: lowered.top_k,
                    };
                    dense_router(&lowered, shared, experts)
                })
                .transpose()?,
            feed_forward_input: match semantic
                .decoder
                .layers
                .get(layer)
                .map(|layer| layer.norm_placement)
            {
                Some(models::semantic::NormPlacement::Parallel) => {
                    crate::FeedForwardInput::LayerInput
                },
                _ => crate::FeedForwardInput::AttentionResidual,
            },
        })
    }
}

/// Session-independent policy used to prepare one NVFP4 routed-MoE layer.
#[derive(Clone, Copy, Debug)]
pub struct NvFp4MoeLayerLoadConfig {
    pub cache: CacheConfig,
    pub max_sequence_blocks: usize,
}

/// Session-independent policy for an affine hybrid linear/full-attention model.
#[derive(Clone, Copy, Debug)]
pub struct SharedRoutedModelLoadConfig {
    pub cache: CacheConfig,
    pub max_sequence_blocks: usize,
    /// Loads the checkpoint's multi-token prediction module for drafting.
    pub speculation: Option<runtime::speculation::Speculation>,
}

impl NvFp4MoeLayerLoadConfig {
    pub(super) fn block(
        self,
        decoder: &DecoderConfig,
        (semantic, plan): (&SemanticModelSpec, &CudaDecoderPlan),
        layer: usize,
    ) -> Result<DecodeMoeBlockConfig> {
        validate_common(decoder, layer, self.max_sequence_blocks)?;
        // Gemma 4's layers run a dense MLP beside their routed experts.
        let (dense_intermediate, routed) = match feed_forward(semantic, layer)? {
            FeedForwardSpec::DenseAndRouted { dense_intermediate_size, routed, .. } => {
                (*dense_intermediate_size, routed)
            },
            FeedForwardSpec::Routed { routed, .. } => (decoder.intermediate_size, routed),
            _ => return Err(unsupported("a decode MoE layer has no routed experts")),
        };
        let attention = attention_config(
            decoder,
            semantic,
            layer,
            self.cache,
            self.max_sequence_blocks,
            crate::ProjectionFormat::Bf16,
        )?;
        Ok(DecodeMoeBlockConfig {
            attention,
            dense_intermediate,
            expert_intermediate: routed.intermediate_size,
            experts: routed.expert_count,
            top_k: routed.top_k,
            router_norm_multiplier: attention_scale(decoder.hidden_size)?,
            activation: GatedActivation::try_from(experts(lowered(plan, layer)?)?.activation)?,
        })
    }
}

/// Attention geometry from the checkpoint configuration; scale and Q/K/V
/// normalization from the layer's semantic spec.
fn attention_config(
    decoder: &DecoderConfig,
    semantic: &SemanticModelSpec,
    layer: usize,
    cache: CacheConfig,
    max_sequence_blocks: usize,
    projection_format: crate::ProjectionFormat,
) -> Result<DecodeAttentionConfig> {
    let spec = softmax_attention(semantic, layer)?;
    let head_dim = decoder.layer_head_dim(layer);
    Ok(DecodeAttentionConfig {
        layer,
        hidden_size: decoder.hidden_size,
        query_heads: decoder.num_attention_heads,
        rotary: super::rotary::embedding(spec, head_dim)?,
        rms_norm_epsilon: to_f32("RMS norm epsilon", decoder.rms_norm_eps)?,
        stream_norm: semantic
            .decoder
            .layers
            .get(layer)
            .map_or(semantic.decoder.final_norm.kind, |spec| spec.input_norm.kind)
            .into(),
        attention_scale: to_f32("attention scale", spec.scale)?,
        projection_format,
        qkv_normalization: spec.qk_normalization.into(),
        window: spec.window,
        score_softcap: spec
            .logit_softcap
            .map(|cap| crate::kernels::ScoreSoftcap::new(to_f32("attention score cap", cap)?))
            .transpose()?,
        query_temperature: spec
            .query_temperature
            .map(crate::kernels::QueryTemperatureScale::new)
            .transpose()?,
        max_sequence_blocks,
        cache: KvStorageSpec::new(cache, decoder.layer_key_value_heads(layer), head_dim),
    })
}

fn softmax_attention(semantic: &SemanticModelSpec, layer: usize) -> Result<&AttentionSpec> {
    match semantic.decoder.layers.get(layer).map(|spec| &spec.mixer) {
        Some(MixerSpec::SoftmaxAttention(attention)) => Ok(attention),
        _ => Err(unsupported("layer is not softmax attention in the model spec")),
    }
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
    if max_sequence_blocks == 0 {
        return Err(unsupported("max_sequence_blocks must be greater than zero"));
    }
    Ok(())
}

fn attention_scale(dimension: usize) -> Result<f32> {
    Ok(to_f32("attention dimension", to_f64(dimension)?)?.sqrt().recip())
}

pub(super) fn to_f64(value: usize) -> Result<f64> {
    let value = u32::try_from(value)?;
    Ok(f64::from(value))
}

pub(super) fn to_f32(label: &str, value: f64) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(unsupported(format!("{label} is outside finite f32 range")));
    }
    #[allow(clippy::cast_possible_truncation)]
    let value = value as f32;
    Ok(value)
}

pub(super) fn unsupported(message: impl Into<String>) -> Error {
    Error::UnsupportedDecoderLayer(message.into())
}
