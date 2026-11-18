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
    pub(super) attention_scale: f32,
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
    /// How the layer's norms scale: by their weight, or by `1 + weight`.
    pub(super) norm: NormalizationLowering,
}

impl DenseSwiGluLayerConfig {
    pub(super) fn from_decoder(decoder: &DecoderConfig, layer: usize) -> Result<Self> {
        if decoder.shared_expert_intermediate_size.is_some() {
            return Err(Error::InvalidModel(
                "dense SwiGLU path does not run shared experts".into(),
            ));
        }
        let sliding = decoder.layer_type(layer) == AttentionLayerType::Sliding;
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
        let config = Self {
            heads: i32::try_from(decoder.num_attention_heads)?,
            kv_heads: i32::try_from(decoder.layer_key_value_heads(layer))?,
            head_dim,
            attention_scale,
            rope_base: decoder
                .rope_theta_for_layer(layer)
                .unwrap_or(10_000.0)
                .to_string()
                .parse()?,
            rope_scaling,
            rope_position_scale,
            rope_attention_factor: rope_scaling
                .and_then(RopeScaling::yarn)
                .map_or(Ok(1.0), |values| values.4.to_string().parse())?,
            rope_traditional: decoder.query_key_rows_interleaved,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            window: decoder.layer_sliding_window(layer),
            activation: GateActivation::from_name(decoder.hidden_activation.as_deref())?,
            top_k_experts: decoder.num_experts.and(decoder.top_k_experts),
            norm: NormalizationLowering::Rms,
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
        NormalizationLowering::Rms => 0.0,
        NormalizationLowering::RmsUnitOffset => 1.0,
    }
}
