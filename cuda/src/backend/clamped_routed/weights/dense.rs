use models::weights::RoutedDecoderLayerBindings;

use super::{ClampedRoutedConfig, ClampedRoutedExpertWeights, tensor};
use crate::{
    CudaBackend, CudaTensorSet, Result,
    backend::{
        clamped_routed::projection::{ClampedRoutedLinearWeight, ClampedRoutedQkvProjections},
        linear::DenseExpertWeights,
    },
};

pub(super) fn load(
    backend: &CudaBackend,
    config: ClampedRoutedConfig,
    tensors: &CudaTensorSet,
    bindings: RoutedDecoderLayerBindings<'_>,
) -> Result<(
    ClampedRoutedQkvProjections,
    ClampedRoutedLinearWeight,
    ClampedRoutedLinearWeight,
    ClampedRoutedExpertWeights,
)> {
    let q = tensor(tensors, &bindings.query.source)?;
    let k = tensor(tensors, &bindings.key.source)?;
    let v = tensor(tensors, &bindings.value.source)?;
    let experts = DenseExpertWeights::load(
        backend,
        tensors,
        bindings.experts,
        config.experts,
        config.hidden,
        config.intermediate,
    )?;
    Ok((
        ClampedRoutedQkvProjections::Native(backend.pack_bf16_linears([&q, &k, &v])?),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.attention_output.source)?),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.router.source)?),
        ClampedRoutedExpertWeights::Dense(Box::new(experts)),
    ))
}
