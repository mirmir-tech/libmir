//! The routed expert banks of a shared-expert `MoE`, by checkpoint storage.

use models::weights::{RoutedExpertBindings, TensorCatalog, TensorStorage};

use super::{
    AffineRoutedMoeWeights, RoutedSharedMoeWeights, nvfp4,
    storage::{routed_is_dense, routed_is_mxfp4, routed_is_mxfp8},
};
use crate::{
    AffineQuantizedWeight, CudaBackend, CudaTensorSet, Error, Result,
    backend::{
        dense::GgmlExpertWeights,
        linear::{DenseExpertWeights, MxFp4ExpertWeights, MxFp8ExpertWeights},
    },
};

/// GGML banks first (they bind like stacked dense ones), then affine,
/// dense, MXFP4, MXFP8 and per-expert NVFP4 storage.
pub(super) fn load(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    bindings: RoutedExpertBindings<'_>,
    (experts, hidden, intermediate): (usize, usize, usize),
) -> Result<RoutedSharedMoeWeights> {
    if let Some(ggml) = GgmlExpertWeights::load(tensors, bindings, (experts, hidden, intermediate))?
    {
        return Ok(RoutedSharedMoeWeights::Ggml(Box::new(ggml)));
    }
    Ok(match bindings {
        RoutedExpertBindings::SeparateGateUp { gate, up, down }
            if matches!(gate.storage, TensorStorage::AffineQuantized { .. })
                && matches!(up.storage, TensorStorage::AffineQuantized { .. })
                && matches!(down.storage, TensorStorage::AffineQuantized { .. }) =>
        {
            RoutedSharedMoeWeights::Affine(Box::new(AffineRoutedMoeWeights {
                gate: AffineQuantizedWeight::load_binding(tensors, gate)?,
                up: AffineQuantizedWeight::load_binding(tensors, up)?,
                down: AffineQuantizedWeight::load_binding(tensors, down)?,
            }))
        },
        routed if routed_is_dense(routed) => RoutedSharedMoeWeights::Dense(Box::new(
            DenseExpertWeights::load(backend, tensors, routed, experts, hidden, intermediate)?,
        )),
        routed if routed_is_mxfp4(routed) => RoutedSharedMoeWeights::MxFp4(Box::new(
            MxFp4ExpertWeights::load(tensors, routed, experts, hidden, intermediate)?,
        )),
        routed if routed_is_mxfp8(routed) => RoutedSharedMoeWeights::MxFp8(Box::new(
            MxFp8ExpertWeights::load(tensors, routed, experts, hidden, intermediate)?,
        )),
        RoutedExpertBindings::Individual { .. } => RoutedSharedMoeWeights::NvFp4(Box::new(
            nvfp4::load(backend, catalog, bindings, experts, hidden, intermediate)?,
        )),
        _ => {
            return Err(Error::UnsupportedDecoderLayer(
                "CUDA shared-routed experts mix incompatible checkpoint storage".into(),
            ));
        },
    })
}
