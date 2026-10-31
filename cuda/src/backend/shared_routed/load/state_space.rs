//! Layers of a state-space hybrid decoder (Nemotron-H): Mamba-2 or
//! attention mixers, each with an optional latent routed feed-forward.

use models::{
    layout::DecoderConfig,
    semantic::{DecoderLayerSpec, FeedForwardSpec, MixerSpec, PositionEncodingSpec},
    weights::{StateSpaceHybridLayerBindings, StateSpaceHybridMixerBindings, TensorCatalog},
};
use runtime::kv::CacheConfig;

use super::{super::SharedRoutedLayerTemplate, required_norm};
use crate::{
    AttentionLayerPolicy, CudaAttentionHybridLayer, CudaBackend, CudaLatentMoe,
    CudaMamba2HybridLayer, CudaTensorSet, Error, LatentMoeConfig, LatentMoeWeights,
    ProjectionFormat, Result,
    backend::{feed_forward::FeedForward, state_space_hybrid::NormedFeedForward},
};

/// Model-wide inputs every hybrid layer is built from.
pub(in crate::backend::shared_routed) struct HybridLoad<'a> {
    pub(in crate::backend::shared_routed) backend: &'a CudaBackend,
    pub(in crate::backend::shared_routed) decoder: &'a DecoderConfig,
    pub(in crate::backend::shared_routed) tensors: &'a CudaTensorSet,
    pub(in crate::backend::shared_routed) catalog: &'a TensorCatalog,
    pub(in crate::backend::shared_routed) cache: CacheConfig,
    pub(in crate::backend::shared_routed) max_sequence_blocks: usize,
}

impl HybridLoad<'_> {
    pub(in crate::backend::shared_routed) fn layer(
        &self,
        layer: usize,
        spec: &DecoderLayerSpec,
        bindings: StateSpaceHybridLayerBindings<'_>,
    ) -> Result<SharedRoutedLayerTemplate> {
        let hidden = self.decoder.hidden_size;
        let feed_forward = bindings
            .feed_forward
            .map(|routed| {
                let hybrid = self.decoder.state_space.as_ref().ok_or_else(|| {
                    Error::UnsupportedDecoderLayer("missing parsed state-space geometry".into())
                })?;
                let config = LatentMoeConfig::from_hybrid(hidden, &hybrid.routed);
                let weights = LatentMoeWeights::load_bindings(
                    self.backend, self.tensors, self.catalog, routed, config,
                )?;
                Ok::<_, Error>(NormedFeedForward {
                    norm: required_norm(self.tensors, &routed.norm.source, hidden)?,
                    block: FeedForward::LatentRouted(Box::new(CudaLatentMoe::new(
                        self.backend, config, weights,
                    )?)),
                })
            })
            .transpose()?;
        match bindings.mixer {
            StateSpaceHybridMixerBindings::StateSpace(mixer) => CudaMamba2HybridLayer::load(
                self.backend,
                self.tensors,
                spec,
                bindings.input_norm,
                mixer,
                feed_forward,
                hidden,
            )
            .map(|layer| SharedRoutedLayerTemplate::Mamba2(Box::new(layer))),
            StateSpaceHybridMixerBindings::Attention(projections) => {
                CudaAttentionHybridLayer::load(
                    self.backend,
                    self.tensors,
                    layer,
                    spec,
                    bindings.input_norm,
                    projections,
                    feed_forward,
                    AttentionLayerPolicy {
                        hidden_size: hidden,
                        projection_format: ProjectionFormat::DirectFp8,
                        cache: self.cache,
                        max_sequence_blocks: self.max_sequence_blocks,
                    },
                )
                .map(|layer| SharedRoutedLayerTemplate::Attention(Box::new(layer)))
            },
        }
    }
}

/// A state-space hybrid stack: Mamba-2 and attention without positional
/// encoding as mixers, latent routed experts or nothing as feed-forwards.
pub(in crate::backend::shared_routed) fn validate_composition(
    layers: &[DecoderLayerSpec],
    max_sequence_blocks: usize,
) -> Result<()> {
    let compatible =
        layers.iter().all(|layer| {
            let mixer = match &layer.mixer {
                MixerSpec::StateSpace(_) => true,
                MixerSpec::SoftmaxAttention(attention) => {
                    attention.position == PositionEncodingSpec::None
                },
                MixerSpec::LinearAttention(_) => false,
            };
            mixer
                && matches!(
                    layer.feed_forward,
                    FeedForwardSpec::LatentRouted { .. } | FeedForwardSpec::None
                )
        }) && layers.iter().any(|layer| matches!(layer.mixer, MixerSpec::SoftmaxAttention(_)));
    if !compatible || max_sequence_blocks == 0 {
        return Err(Error::UnsupportedDecoderLayer(
            "parsed decoder is not a state-space hybrid stack".into(),
        ));
    }
    Ok(())
}
