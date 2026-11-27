//! The mixer of a hybrid layer: gated `DeltaNet`, gated full attention or
//! multi-head latent attention.

use models::{
    layout::DecoderConfig,
    semantic::{DecoderLayerSpec, MixerSpec},
    weights::{HybridMixerBindings, LatentAttentionBindings},
};

use crate::engine::{
    Array, DecoderCache, Error, GatedDeltaLayer, GatedDeltaLayerConfig, GatedFullAttention,
    GatedFullAttentionConfig, ModelTensors, Result, Stream,
    decoder::{LayerContext, MixerKind},
    latent_attention::LatentAttention,
    paged_attention_min_context,
};

#[derive(Debug)]
pub(in super::super) enum Attention {
    Linear(GatedDeltaLayer),
    Full(GatedFullAttention),
    Latent(Box<LatentAttention>),
}

impl Attention {
    pub(super) fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        layer: &DecoderLayerSpec,
        bindings: HybridMixerBindings<'_>,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let epsilon = layer.input_norm.epsilon;
        match (bindings, &layer.mixer) {
            (HybridMixerBindings::Linear(bindings), MixerSpec::LinearAttention(linear)) => {
                let layout =
                    decoder.linear_attention.as_ref().map(|linear| linear.layout).ok_or_else(
                        || Error::InvalidModel("missing linear attention configuration".into()),
                    )?;
                let config = GatedDeltaLayerConfig::from_spec(linear, layout, epsilon)?;
                GatedDeltaLayer::load_bindings(tensors, bindings, config, stream).map(Self::Linear)
            },
            (HybridMixerBindings::Softmax(bindings), MixerSpec::SoftmaxAttention(attention)) => {
                GatedFullAttention::load_bindings(
                    tensors,
                    bindings,
                    GatedFullAttentionConfig::from_spec(attention, epsilon)?,
                    norm_shift,
                    stream,
                )
                .map(Self::Full)
            },
            _ => Err(Error::InvalidModel(format!(
                "layer {}'s mixer bindings do not match its specification",
                layer.index
            ))),
        }
    }

    pub(super) fn load_latent(
        tensors: &ModelTensors,
        layer: &DecoderLayerSpec,
        bindings: LatentAttentionBindings<'_>,
        stream: &Stream,
    ) -> Result<Self> {
        let MixerSpec::LatentAttention(spec) = &layer.mixer else {
            return Err(Error::InvalidModel(format!(
                "layer {} is not specified as latent attention",
                layer.index
            )));
        };
        LatentAttention::load(tensors, spec, bindings, stream)
            .map(|layer| Self::Latent(Box::new(layer)))
    }

    pub(super) const fn mixer_kind(&self) -> MixerKind {
        match self {
            Self::Linear(_) => MixerKind::Linear,
            Self::Full(_) | Self::Latent(_) => MixerKind::Softmax,
        }
    }

    pub(super) const fn kind(&self) -> &'static str {
        match self {
            Self::Linear(_) => "gated_delta",
            Self::Full(_) => "gated_full",
            Self::Latent(_) => "latent",
        }
    }

    /// Mixes normalized `[1, sequence, hidden]` states over layer `index`'s
    /// cache.
    pub(super) fn forward(
        &self,
        normalized: &Array,
        cache: &mut DecoderCache,
        index: usize,
        context: LayerContext<'_>,
    ) -> Result<Array> {
        let LayerContext { position, causal, positions, stream, .. } = context;
        match self {
            Self::Linear(layer) => {
                layer.forward(normalized, cache.gated_delta_state(index)?, stream)
            },
            Self::Full(layer) => layer.forward_with_positions(
                normalized,
                cache.full_attention_cache(index)?,
                paged_attention_min_context(stream),
                position,
                causal,
                positions,
                stream,
            ),
            Self::Latent(layer) => layer.forward(
                normalized,
                cache.full_attention_cache(index)?,
                position,
                causal,
                stream,
            ),
        }
    }
}
