//! Multi-head latent attention stacks: every
//! layer is latent attention followed by a dense or shared-routed
//! feed-forward.

use models::{
    layout::DecoderConfig,
    semantic::{MixerSpec, SemanticModelSpec},
    weights::{TensorCatalog, WeightBindingPlan},
};
use runtime::kv::CacheConfig;

use super::super::SharedRoutedLayerTemplate;
use crate::{CudaBackend, CudaLatentAttentionLayer, CudaTensorSet, Error, Result};

pub(in super::super) fn validate_composition(
    semantic: &SemanticModelSpec,
    max_sequence_blocks: usize,
) -> Result<()> {
    let latent = semantic
        .decoder
        .layers
        .iter()
        .all(|layer| matches!(layer.mixer, MixerSpec::LatentAttention(_)));
    if !latent || semantic.decoder.layers.is_empty() || max_sequence_blocks == 0 {
        return Err(Error::UnsupportedDecoderLayer(
            "parsed decoder is not a latent-attention stack".into(),
        ));
    }
    Ok(())
}

pub(in super::super) fn layers(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    semantic: &SemanticModelSpec,
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    bindings: &WeightBindingPlan,
    cache: CacheConfig,
) -> Result<Vec<SharedRoutedLayerTemplate>> {
    semantic
        .decoder
        .layers
        .iter()
        .map(|spec| {
            CudaLatentAttentionLayer::load(
                backend,
                decoder,
                tensors,
                catalog,
                spec,
                bindings.latent_decoder_layer(spec.index)?,
                cache,
            )
            .map(|layer| SharedRoutedLayerTemplate::Latent(Box::new(layer)))
        })
        .collect()
}
