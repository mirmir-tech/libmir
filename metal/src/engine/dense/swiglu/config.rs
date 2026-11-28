use models::{
    layout::DecoderConfig,
    semantic::{
        AttentionSpec, AttentionWindow, DecoderLayerSpec, DecoderSpec, FeedForwardSpec, MixerSpec,
        PositionEncodingSpec, QueryTemperature,
    },
};

pub(super) use crate::engine::mlp_gate::GateActivation;
use crate::engine::{Error, Result, lowering::NormalizationLowering};

#[derive(Debug, Clone, Copy)]
pub(super) struct DenseSwiGluLayerConfig {
    pub(super) heads: i32,
    pub(super) kv_heads: i32,
    pub(super) head_dim: i32,
    /// Leading features of each head the rotation pairs over (the whole head
    /// for proportional `RoPE`, whose unrotated pairs pass through); zero
    /// without `RoPE`.
    pub(super) rope_dimensions: i32,
    pub(super) attention_scale: f32,
    /// Cap on attention scores before the softmax (Gemma 2).
    pub(super) attention_softcap: Option<f32>,
    /// Unscaled `RoPE`'s base; scaled `RoPE` rotates by the weights'
    /// frequencies.
    pub(super) rope_base: f32,
    /// The fused kernel's position scale: `1 / factor` under linear scaling.
    pub(super) rope_position_scale: f32,
    /// `YaRN`'s and `LongRoPE`'s factor on the rotated pairs.
    pub(super) rope_attention_factor: f32,
    /// Rotate adjacent pairs instead of halves: llama.cpp's converter stores
    /// Llama-family Q and K rows interleaved, and pairwise rotation of those
    /// rows gives the same attention as half rotation of the source rows.
    pub(super) rope_traditional: bool,
    pub(super) rms_norm_eps: f32,
    /// Which earlier keys a query sees: a sliding window or its chunk.
    pub(super) window: Option<AttentionWindow>,
    pub(super) activation: GateActivation,
    /// Query temperature (Llama 4's unrotated layers).
    pub(super) temperature: Option<QueryTemperature>,
    /// How the layer's norms scale: by their weight, or by `1 + weight`.
    pub(super) norm: NormalizationLowering,
    /// Factor on the attention and feed-forward outputs before each residual
    /// join (Granite).
    pub(super) residual_multiplier: Option<f32>,
}

impl DenseSwiGluLayerConfig {
    /// The chunk of a chunked layer (Llama 4's rotary layers).
    pub(super) const fn chunk(&self) -> Option<usize> {
        match self.window {
            Some(AttentionWindow::Chunked { chunk }) => Some(chunk.get()),
            Some(AttentionWindow::Sliding(_)) | None => None,
        }
    }

    /// The layer's configuration from its semantic specification;
    /// `interleaved_rows` is llama.cpp's Q/K row order, a property of the
    /// stored weights.
    pub(super) fn from_spec(
        decoder: &DecoderSpec,
        layer: &DecoderLayerSpec,
        interleaved_rows: bool,
    ) -> Result<Self> {
        let attention = attention(layer)?;
        let head_dim = i32::try_from(attention.head_dim)?;
        let (rope_dimensions, rope_base, rope_attention_factor) = match &attention.position {
            PositionEncodingSpec::Rotary(rotary) => (
                i32::try_from(rotary.pairing_dimensions(attention.head_dim)?)?,
                decimal(rotary.theta)?,
                attention
                    .position
                    .inverse_frequencies(rotary.pairing_dimensions(attention.head_dim)?)?
                    .map_or(Ok(1.0), |(_, factor)| decimal(factor))?,
            ),
            PositionEncodingSpec::None => (0, 10_000.0, 1.0),
        };
        let config = Self {
            heads: i32::try_from(attention.query_heads)?,
            kv_heads: i32::try_from(attention.key_value_heads)?,
            head_dim,
            rope_dimensions,
            attention_scale: decimal(attention.scale)?,
            attention_softcap: attention.logit_softcap.map(decimal).transpose()?,
            rope_base,
            rope_position_scale: crate::engine::attention::rope_position_scale(attention)?,
            rope_attention_factor,
            rope_traditional: interleaved_rows,
            rms_norm_eps: decimal(layer.input_norm.epsilon)?,
            window: attention.window,
            activation: activation(&layer.feed_forward)?,
            temperature: attention.query_temperature,
            norm: NormalizationLowering::Rms,
            residual_multiplier: (decoder.residual_multiplier.to_bits() != 1.0_f64.to_bits())
                .then(|| decimal(decoder.residual_multiplier))
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
    use crate::engine::lowering::{Activation, Experts, FeedForwardLowering, MixerLowering};
    let compatible = lowering.len() == decoder.num_hidden_layers
        && lowering.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                FeedForwardLowering::Dense {
                    activation: Activation::SwiGlu { clamp: None } | Activation::GeluTanh,
                } | FeedForwardLowering::Routed {
                    experts: Experts {
                        activation: Activation::SwiGlu { clamp: None },
                        ..
                    },
                    ..
                }
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
        self.chunk().is_some() || self.temperature.is_some()
    }

    /// Whether attention runs per row over the chronological K/V view:
    /// capped scores, chunked keys or a position-dependent query scale,
    /// which the persistent and packed batch paths do not apply.
    pub(super) const fn attends_per_row(self) -> bool {
        self.attention_softcap.is_some() || self.positional()
    }
}

/// The layer's softmax attention.
pub(super) fn attention(layer: &DecoderLayerSpec) -> Result<&AttentionSpec> {
    match &layer.mixer {
        MixerSpec::SoftmaxAttention(attention) => Ok(attention),
        _ => Err(Error::InvalidModel(format!(
            "dense layer {} requires softmax attention",
            layer.index
        ))),
    }
}

/// The gate activation of the layer's gated MLP or routed experts.
fn activation(feed_forward: &FeedForwardSpec) -> Result<GateActivation> {
    match feed_forward {
        FeedForwardSpec::Dense { activation, .. }
        | FeedForwardSpec::DenseAndRouted { dense_activation: activation, .. } => {
            GateActivation::from_spec(activation)
        },
        FeedForwardSpec::Routed { routed, .. } | FeedForwardSpec::LatentRouted { routed, .. } => {
            GateActivation::from_spec(&routed.activation)
        },
        FeedForwardSpec::None => Ok(GateActivation::Silu),
    }
}

/// A semantic value in the runtime's single precision.
fn decimal(value: f64) -> Result<f32> {
    Ok(value.to_string().parse()?)
}
