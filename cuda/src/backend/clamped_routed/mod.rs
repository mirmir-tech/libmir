mod capability;
mod config;
mod layer;
mod layout;
mod load;
mod plan;
mod projection;
mod scratch;
mod session;
mod validation;
mod weights;

use models::layout::DecoderConfig;
use runtime::kv::CacheConfig;

pub use self::session::CudaClampedRoutedModelSession;
use self::{
    capability::{ClampedRoutedCapabilityPlan, ClampedRoutedQkvLowering},
    config::ClampedRoutedConfig,
    layer::ClampedRoutedLayerTemplate,
    layout::ClampedRoutedLayout,
    projection::ClampedRoutedBoundaryProjection,
};
use crate::{CudaBackend, CudaTensor, Result};

#[derive(Clone)]
pub struct CudaClampedRoutedModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: ClampedRoutedBoundaryProjection,
    final_norm: CudaTensor,
    output: ClampedRoutedBoundaryProjection,
    layers: Vec<ClampedRoutedLayerTemplate>,
    config: ClampedRoutedConfig,
    cache: CacheConfig,
    max_sequence_blocks: usize,
}

impl CudaClampedRoutedModelTemplate {
    #[must_use]
    pub const fn decoder(&self) -> &DecoderConfig {
        &self.decoder
    }

    pub fn instantiate(&self) -> Result<CudaClampedRoutedModelSession> {
        CudaClampedRoutedModelSession::new(self)
    }
}
