mod batch;
mod cache;
mod diagnostics;
mod fusion;

pub use cache::MetalCacheConfig;
pub use diagnostics::MetalDiagnosticsConfig;
pub use fusion::{FeatureToggle, FusionMode, MetalFusionConfig};

#[derive(Debug, Clone)]
pub struct MetalConfig {
    pub batch: MetalBatchConfig,
    pub cache: MetalCacheConfig,
    pub diagnostics: MetalDiagnosticsConfig,
    pub fusion: MetalFusionConfig,
    pub kv_cache: runtime::kv::CacheConfig,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            batch: MetalBatchConfig::default(),
            cache: MetalCacheConfig::default(),
            diagnostics: MetalDiagnosticsConfig::default(),
            fusion: MetalFusionConfig::default(),
            kv_cache: runtime::kv::CacheConfig::new(4_096),
        }
    }
}
pub use batch::{DenseBatchMode, MetalBatchConfig};
