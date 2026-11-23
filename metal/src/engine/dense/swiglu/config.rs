use models::layout::{AttentionLayerType, DecoderConfig, RopeScaling};

use crate::engine::{Error, Result, lowering::NormalizationLowering};

/// The gate activation of the layer's gated MLP (`act(gate) * up`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GateActivation {
    /// `SiLU`: Qwen, Llama, Mistral.
    Silu,
    /// Tanh-approximated GELU: Gemma.
    GeluTanh,
}

impl GateActivation {
    fn from_name(name: Option<&str>) -> Result<Self> {
        match name {
            None | Some("silu") => Ok(Self::Silu),
            Some("gelu_pytorch_tanh" | "gelu_tanh") => Ok(Self::GeluTanh),
            Some(other) => {
                Err(Error::InvalidModel(format!("dense Metal MLP has no {other} gate activation")))
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DenseSwiGluLayerConfig {
    pub(super) heads: i32,
    pub(super) kv_heads: i32,
    pub(super) head_dim: i32,
    /// Leading features of each head that rotate (`partial_rotary_factor`).
    pub(super) rope_dimensions: i32,
    pub(super) attention_scale: f32,
    /// Cap on attention scores before the softmax (Gemma 2).
    pub(super) attention_softcap: Option<f32>,
    pub(super) rope_base: f32,
    pub(super) rope_scaling: Option<RopeScaling>,
    /// `RoPE` position scale for MLX: `1 / factor` under linear scaling, else
    /// 1.
    pub(super) rope_position_scale: f32,
    pub(super) rope_attention_factor: f32,
    /// Rotate adjacent pairs instead of halves: llama.cpp's converter stores
    /// Llama-family Q and K rows interleaved, and pairwise rotation of those
    /// rows gives the same attention as half rotation of the source rows.
    pub(super) rope_traditional: bool,
    pub(super) rms_norm_eps: f32,
    /// Attention span of a sliding layer, in tokens.
    pub(super) window: Option<usize>,
    pub(super) activation: GateActivation,
    /// Experts each token selects, for layers with routed experts.
    pub(super) top_k_experts: Option<usize>,
    /// Chunked attention (Llama 4's rotary layers).
    pub(super) chunk: Option<usize>,
    /// Query temperature (Llama 4's unrotated layers).
    pub(super) temperature: Option<models::semantic::QueryTemperature>,
    /// Sparsemixer jitter of routed layers (Phi-3.5-MoE).
    pub(super) sparse_mixer_jitter: Option<f32>,
    /// How the layer's norms scale: by their weight, or by `1 + weight`.
    pub(super) norm: NormalizationLowering,
    /// Factor on the attention and feed-forward outputs before each residual
    /// join (Granite).
    pub(super) residual_multiplier: Option<f32>,
}

impl DenseSwiGluLayerConfig {
    /// The recent tokens the layer's cache keeps: a sliding window, or a
    /// chunked layer's chunk, which holds every key its queries may see.
    pub(super) const fn span(&self) -> Option<usize> {
        match self.window {
            Some(window) => Some(window),
            None => self.chunk,
        }
    }

    pub(super) fn from_decoder(decoder: &DecoderConfig, layer: usize) -> Result<Self> {
        let sliding = decoder.layer_type(layer) == AttentionLayerType::Sliding;
        let rotary_layer = decoder.attention_pattern.rotates(layer);
        // Gemma 3 scales RoPE only on its global layers.
        let rope_scaling = decoder
            .rope_scaling
            .filter(|_| !(sliding && decoder.sliding_attention_rope_unscaled));
        let head_dim = i32::try_from(decoder.layer_head_dim(layer))?;
        let attention_scale = match decoder.attention_scale {
            Some(scale) => scale.to_string().parse::<f32>()?,
            None => head_dim.to_string().parse::<f32>()?.sqrt().recip(),
        };
        let rope_position_scale = match rope_scaling {
            Some(RopeScaling::Linear { factor }) => factor.to_string().parse::<f32>()?.recip(),
            _ => 1.0,
        };
        // Rotation-free layers (Cohere 2's global, Llama 4's every fourth) rotate
        // nothing.
        let rope_dimensions = match decoder.partial_rotary_factor_for_layer(layer) {
            _ if !decoder.attention_pattern.rotates(layer) => 0,
            Some(factor) if factor.to_bits() != 1.0_f64.to_bits() => {
                let rotated = (f64::from(head_dim) * factor).round();
                format!("{rotated}").parse::<i32>().unwrap_or(0)
            },
            _ => head_dim,
        };
        let config = Self {
            heads: i32::try_from(decoder.num_attention_heads)?,
            kv_heads: i32::try_from(decoder.layer_key_value_heads(layer))?,
            head_dim,
            rope_dimensions,
            attention_scale,
            attention_softcap: decoder
                .attn_logit_softcapping
                .map(|cap| cap.to_string().parse::<f32>())
                .transpose()?,
            rope_base: decoder
                .rope_theta_for_layer(layer)
                .unwrap_or(10_000.0)
                .to_string()
                .parse()?,
            rope_scaling,
            rope_position_scale,
            rope_attention_factor: rope_scaling
                .map_or(1.0, RopeScaling::attention_factor)
                .to_string()
                .parse()?,
            rope_traditional: decoder.query_key_rows_interleaved,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            window: decoder.layer_sliding_window(layer),
            activation: GateActivation::from_name(decoder.hidden_activation.as_deref())?,
            top_k_experts: decoder.num_experts.and(decoder.top_k_experts),
            chunk: decoder.attention_pattern.rotary_chunk.filter(|_| rotary_layer),
            temperature: decoder
                .attention_pattern
                .rotation_free_temperature
                .filter(|_| !rotary_layer),
            sparse_mixer_jitter: match decoder.expert_router {
                models::layout::ExpertRouter::SparseMixer { jitter } => {
                    Some(jitter.to_string().parse()?)
                },
                models::layout::ExpertRouter::SoftmaxTopK
                | models::layout::ExpertRouter::SigmoidInputScaled => None,
            },
            norm: NormalizationLowering::Rms,
            residual_multiplier: decoder
                .residual_multiplier
                .filter(|multiplier| multiplier.to_bits() != 1.0_f64.to_bits())
                .map(|multiplier| multiplier.to_string().parse::<f32>())
                .transpose()?,
        };
        config.validate()?;
        Ok(config)
    }

    /// The constant added to every norm weight of the layer.
    pub(super) const fn norm_shift(self) -> f32 {
        norm_shift(self.norm)
    }

    fn validate(self) -> Result<()> {
        if [self.heads, self.kv_heads, self.head_dim]
            .into_iter()
            .any(|dimension| dimension <= 0)
            || self.rope_dimensions < 0
            || self.rope_dimensions > self.head_dim
            || self.rope_dimensions % 2 != 0
        {
            return Err(Error::InvalidModel(format!(
                "non-positive dense SwiGLU dimensions: {self:?}"
            )));
        }
        if !self.rope_base.is_finite()
            || !self.rms_norm_eps.is_finite()
            || !self.attention_scale.is_finite()
            || !self.rope_position_scale.is_finite()
            || !self.rope_attention_factor.is_finite()
            || self.rope_attention_factor <= 0.0
        {
            return Err(Error::InvalidModel(format!("non-finite dense SwiGLU config: {self:?}")));
        }
        Ok(())
    }
}

/// The constant a norm adds to its learned weight.
pub(super) const fn norm_shift(norm: NormalizationLowering) -> f32 {
    match norm {
        NormalizationLowering::Rms | NormalizationLowering::Layer => 0.0,
        NormalizationLowering::RmsUnitOffset => 1.0,
    }
}

/// Every layer softmax attention with a gated MLP, routed experts, or routed
/// plus shared experts.
pub(super) fn require_dense_layers(
    decoder: &DecoderConfig,
    lowering: &[crate::engine::lowering::LayerLowering],
) -> Result<()> {
    use crate::engine::lowering::{FeedForwardLowering, MixerLowering};
    let compatible = lowering.len() == decoder.num_hidden_layers
        && lowering.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                FeedForwardLowering::Dense
                    | FeedForwardLowering::Routed
                    | FeedForwardLowering::SharedRouted
            ) && matches!(layer.mixer, MixerLowering::Softmax { .. })
        });
    if compatible {
        Ok(())
    } else {
        Err(Error::InvalidModel(
            "dense SwiGLU loader requires softmax attention with gated or routed MLPs".into(),
        ))
    }
}

impl DenseSwiGluLayerConfig {
    /// Whether keys or the query scale depend on the query's position.
    pub(super) const fn positional(self) -> bool {
        self.chunk.is_some() || self.temperature.is_some()
    }
}
