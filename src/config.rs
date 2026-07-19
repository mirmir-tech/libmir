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
    /// Resource policy applied before vision preprocessing and execution.
    pub vision: VisionRuntimeConfig,
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
            vision: VisionRuntimeConfig::default(),
            #[cfg(feature = "cuda")]
            cuda: CudaConfig::default(),
            #[cfg(feature = "metal")]
            metal: MetalConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Runtime resource limits for image inputs.
pub struct VisionRuntimeConfig {
    /// Maximum resized image area. `None` keeps the checkpoint-declared cap.
    pub max_pixels: Option<usize>,
    /// Maximum bytes reserved for one vision attention score matrix.
    ///
    /// `None` derives the budget from currently available accelerator memory.
    pub attention_budget_bytes: Option<u64>,
    /// Percentage of currently available memory used by the automatic budget.
    pub memory_percent: u8,
}

impl Default for VisionRuntimeConfig {
    fn default() -> Self {
        Self {
            max_pixels: None,
            attention_budget_bytes: None,
            memory_percent: 80,
        }
    }
}
