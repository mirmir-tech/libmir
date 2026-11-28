use std::sync::Arc;

use models::{
    layout::VisionConfig,
    weights::{TensorCatalog, TensorReadiness},
};

use crate::{
    CudaBackend, Result,
    backend::{CudaPixelShuffleVisionTower, CudaPooledVisionTower, CudaSpatialMergeVisionTower},
    checkpoint::load_vision_tensors_with_inline,
};

pub(in crate::engine) enum LoadedVisionModel {
    Pooled(CudaPooledVisionTower),
    SpatialMerge(CudaSpatialMergeVisionTower),
    PixelShuffle(CudaPixelShuffleVisionTower),
}

pub(in crate::engine) fn load_vision_model(
    backend: &CudaBackend,
    config: Option<&VisionConfig>,
    readiness: Option<&TensorReadiness>,
    catalog: &TensorCatalog,
    inline: Option<&(String, Arc<[u8]>)>,
) -> Result<Option<LoadedVisionModel>> {
    if !readiness.is_some_and(TensorReadiness::is_ready) {
        return Ok(None);
    }
    match config {
        Some(VisionConfig::PooledEncoder(config)) => {
            let tensors = load_vision_tensors_with_inline(
                backend,
                &VisionConfig::PooledEncoder(config.clone()),
                catalog,
                inline,
            )?;
            Ok(Some(LoadedVisionModel::Pooled(CudaPooledVisionTower::new(
                backend,
                config.clone(),
                tensors,
            )?)))
        },
        Some(VisionConfig::SpatialMergeEncoder(config)) => {
            let tensors = load_vision_tensors_with_inline(
                backend,
                &VisionConfig::SpatialMergeEncoder(config.clone()),
                catalog,
                inline,
            )?;
            Ok(Some(LoadedVisionModel::SpatialMerge(CudaSpatialMergeVisionTower::new(
                backend,
                config.clone(),
                tensors,
            )?)))
        },
        Some(VisionConfig::PixelShuffleEncoder(config)) => {
            let tensors = load_vision_tensors_with_inline(
                backend,
                &VisionConfig::PixelShuffleEncoder(config.clone()),
                catalog,
                inline,
            )?;
            Ok(Some(LoadedVisionModel::PixelShuffle(CudaPixelShuffleVisionTower::new(
                backend,
                config.clone(),
                tensors,
            )?)))
        },
        None => Ok(None),
    }
}
