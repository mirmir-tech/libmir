//! Layers of a state-space hybrid decoder (Nemotron-H): Mamba-2 or
//! attention mixers, each with an optional latent routed or ungated `relu²`
//! feed-forward.

use architecture::{
    cuda::CudaLayer,
    lowering::{Activation, FeedForwardLowering},
};
use models::{
    layout::{DecoderConfig, StateSpaceLayout},
    semantic::{DecoderLayerSpec, FeedForwardSpec},
    weights::{
        StateSpaceHybridFeedForwardBindings, StateSpaceHybridLayerBindings,
        StateSpaceHybridMixerBindings, TensorCatalog,
    },
};
use runtime::kv::CacheConfig;

use super::{super::SharedRoutedLayerTemplate, required_norm};
use crate::{
    AttentionLayerPolicy, CudaAttentionHybridLayer, CudaBackend, CudaLatentMoe,
    CudaMamba2HybridLayer, CudaTensorSet, Error, LatentMoeConfig, LatentMoeWeights, Result,
    backend::{
        feed_forward::{DenseFeedForward, FeedForward},
        linear::CheckpointProjectionWeight,
        state_space_hybrid::NormedFeedForward,
    },
};

/// Model-wide inputs every hybrid layer is built from.
pub(in crate::backend::shared::routed) struct HybridLoad<'a> {
    pub(in crate::backend::shared::routed) backend: &'a CudaBackend,
    pub(in crate::backend::shared::routed) decoder: &'a DecoderConfig,
    pub(in crate::backend::shared::routed) tensors: &'a CudaTensorSet,
    pub(in crate::backend::shared::routed) catalog: &'a TensorCatalog,
    pub(in crate::backend::shared::routed) cache: CacheConfig,
    pub(in crate::backend::shared::routed) max_sequence_blocks: usize,
}

impl HybridLoad<'_> {
    pub(in crate::backend::shared::routed) fn layer(
        &self,
        layer: usize,
        (spec, lowered): (&DecoderLayerSpec, &CudaLayer),
        bindings: StateSpaceHybridLayerBindings<'_>,
    ) -> Result<SharedRoutedLayerTemplate> {
        let hidden = self.decoder.hidden_size;
        let feed_forward = bindings
            .feed_forward
            .map(|feed_forward| self.feed_forward((spec, lowered), feed_forward))
            .transpose()?;
        match bindings.mixer {
            StateSpaceHybridMixerBindings::StateSpace(mixer) => CudaMamba2HybridLayer::load(
                self.backend,
                self.tensors,
                spec,
                (bindings.input_norm, mixer),
                feed_forward,
                (hidden, self.decay_layout()?),
            )
            .map(|layer| SharedRoutedLayerTemplate::Mamba2(Box::new(layer))),
            StateSpaceHybridMixerBindings::Attention(projections) => {
                let projection_format =
                    CheckpointProjectionWeight::load_binding(self.tensors, projections.query)?
                        .format();
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
                        projection_format,
                        cache: self.cache,
                        max_sequence_blocks: self.max_sequence_blocks,
                    },
                )
                .map(|layer| SharedRoutedLayerTemplate::Attention(Box::new(layer)))
            },
        }
    }
}

impl HybridLoad<'_> {
    fn decay_layout(&self) -> Result<StateSpaceLayout> {
        self.decoder
            .state_space
            .as_ref()
            .map(|hybrid| hybrid.mamba.layout)
            .ok_or_else(|| {
                Error::UnsupportedDecoderLayer("missing parsed Mamba-2 configuration".into())
            })
    }

    /// The block admission lowered, at the geometry the spec declares.
    fn feed_forward(
        &self,
        (spec, lowered): (&DecoderLayerSpec, &CudaLayer),
        bindings: StateSpaceHybridFeedForwardBindings<'_>,
    ) -> Result<NormedFeedForward> {
        let hidden = self.decoder.hidden_size;
        match (bindings, &spec.feed_forward, lowered.feed_forward) {
            (
                StateSpaceHybridFeedForwardBindings::Dense(dense),
                FeedForwardSpec::Dense { intermediate_size, .. },
                FeedForwardLowering::Dense { activation: Activation::ReluSquared },
            ) => Ok(NormedFeedForward {
                norm: required_norm(self.tensors, &dense.norm.source, hidden)?,
                block: FeedForward::Dense(Box::new(DenseFeedForward::load_relu_squared(
                    self.backend,
                    (hidden, *intermediate_size),
                    self.tensors,
                    (dense.up, dense.down),
                )?)),
            }),
            (
                StateSpaceHybridFeedForwardBindings::Routed(routed),
                spec @ FeedForwardSpec::LatentRouted { .. },
                FeedForwardLowering::LatentRouted { .. },
            ) => {
                let config = LatentMoeConfig::from_spec(hidden, spec)?;
                let weights = LatentMoeWeights::load_bindings(
                    self.backend, self.tensors, self.catalog, routed, config,
                )?;
                Ok(NormedFeedForward {
                    norm: required_norm(self.tensors, &routed.norm.source, hidden)?,
                    block: FeedForward::LatentRouted(Box::new(CudaLatentMoe::new(
                        self.backend, config, weights,
                    )?)),
                })
            },
            _ => Err(Error::UnsupportedDecoderLayer(
                "hybrid feed-forward bindings differ from the layer's semantics".into(),
            )),
        }
    }
}
