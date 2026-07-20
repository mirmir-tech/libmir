use models::{
    layout::VisionConfig,
    weights::{TensorCatalog, TensorReadiness},
};

use crate::{
    CudaBackend, Result,
    backend::{CudaPooledVisionTower, CudaSpatialMergeVisionTower},
    checkpoint::load_vision_tensors,
};

pub(super) enum LoadedVisionModel {
    Pooled(CudaPooledVisionTower),
    SpatialMerge(CudaSpatialMergeVisionTower),
}

pub(super) fn load_vision_model(
    backend: &CudaBackend,
    config: Option<&VisionConfig>,
    readiness: Option<&TensorReadiness>,
    catalog: &TensorCatalog,
) -> Result<Option<LoadedVisionModel>> {
    if !readiness.is_some_and(TensorReadiness::is_ready) {
        return Ok(None);
    }
    match config {
        Some(VisionConfig::PooledEncoder(config)) => {
            let tensors = load_vision_tensors(
                backend,
                &VisionConfig::PooledEncoder(config.clone()),
                catalog,
            )?;
            Ok(Some(LoadedVisionModel::Pooled(CudaPooledVisionTower::new(
                backend,
                config.clone(),
                tensors,
            )?)))
        },
        Some(VisionConfig::SpatialMergeEncoder(config)) => {
            let tensors = load_vision_tensors(
                backend,
                &VisionConfig::SpatialMergeEncoder(config.clone()),
                catalog,
            )?;
            Ok(Some(LoadedVisionModel::SpatialMerge(CudaSpatialMergeVisionTower::new(
                backend,
                config.clone(),
                tensors,
            )?)))
        },
        None => Ok(None),
    }
}
