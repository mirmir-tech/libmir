//! The mixer of a hybrid layer: gated `DeltaNet`, gated full attention or
//! multi-head latent attention.

use models::{
    layout::DecoderConfig,
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
        bindings: HybridMixerBindings<'_>,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        match bindings {
            HybridMixerBindings::Linear(bindings) => {
                let config = GatedDeltaLayerConfig::from_linear_attention(
                    decoder.linear_attention.as_ref().ok_or_else(|| {
                        Error::InvalidModel("missing linear attention configuration".into())
                    })?,
                    decoder.rms_norm_eps,
                )?;
                GatedDeltaLayer::load_bindings(tensors, bindings, config, stream).map(Self::Linear)
            },
            HybridMixerBindings::Softmax(bindings) => GatedFullAttention::load_bindings(
                tensors,
                bindings,
                GatedFullAttentionConfig::from_decoder(decoder)?,
                norm_shift,
                stream,
            )
            .map(Self::Full),
        }
    }

    pub(super) fn load_latent(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        index: usize,
        bindings: LatentAttentionBindings<'_>,
        stream: &Stream,
    ) -> Result<Self> {
        LatentAttention::load(tensors, decoder, index, bindings, stream)
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
