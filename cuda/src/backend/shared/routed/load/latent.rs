//! Multi-head latent attention stacks: every
//! layer is latent attention followed by a dense or shared-routed
//! feed-forward.

use architecture::cuda::CudaLayer;
use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, WeightBindingPlan},
};
use runtime::kv::CacheConfig;

use super::super::SharedRoutedLayerTemplate;
use crate::{CudaBackend, CudaLatentAttentionLayer, CudaTensorSet, Result};

pub(in super::super) fn layers(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    (semantic, plan): (&SemanticModelSpec, &[CudaLayer]),
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    bindings: &WeightBindingPlan,
    cache: CacheConfig,
) -> Result<Vec<SharedRoutedLayerTemplate>> {
    semantic
        .decoder
        .layers
        .iter()
        .zip(plan)
        .map(|(spec, lowered)| {
            CudaLatentAttentionLayer::load(
                backend,
                decoder,
                tensors,
                catalog,
                (spec, lowered),
                bindings.latent_decoder_layer(spec.index)?,
                cache,
            )
            .map(|layer| SharedRoutedLayerTemplate::Latent(Box::new(layer)))
        })
        .collect()
}
