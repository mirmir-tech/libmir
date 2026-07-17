#[cfg(feature = "cuda")]
use cuda::CudaConfig;
#[cfg(feature = "metal")]
use metal::MetalConfig;
use runtime::{kv::CacheConfig, scheduler::SchedulerConfig};

#[derive(Debug, Clone)]
/// Configuration shared by model loading, scheduling, and backend execution.
pub struct RuntimeConfig {
    /// K/V cache block layout, capacity, and storage type.
    pub kv_cache: CacheConfig,
    /// Decode batching and scheduling policy.
    pub scheduler: SchedulerConfig,
    #[cfg(feature = "cuda")]
    /// CUDA backend configuration.
    pub cuda: CudaConfig,
    #[cfg(feature = "metal")]
    /// Metal backend configuration.
    pub metal: MetalConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            kv_cache: CacheConfig::new(4_096),
            scheduler: SchedulerConfig::default(),
            #[cfg(feature = "cuda")]
            cuda: CudaConfig::default(),
            #[cfg(feature = "metal")]
            metal: MetalConfig::default(),
        }
    }
}
