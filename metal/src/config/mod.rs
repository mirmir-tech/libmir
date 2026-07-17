mod batch;
mod cache;
mod diagnostics;
mod fusion;

pub use cache::MetalCacheConfig;
pub use diagnostics::MetalDiagnosticsConfig;
pub use fusion::{FeatureToggle, FusionMode, MetalFusionConfig};

#[derive(Debug, Clone, Default)]
pub struct MetalConfig {
    pub batch: MetalBatchConfig,
    pub cache: MetalCacheConfig,
    pub diagnostics: MetalDiagnosticsConfig,
    pub fusion: MetalFusionConfig,
}
pub use batch::{DenseBatchMode, MetalBatchConfig};
