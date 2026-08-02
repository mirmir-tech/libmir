#[cfg(feature = "cuda")]
use cuda::CudaConfig;
#[cfg(feature = "metal")]
use metal::MetalConfig;
use runtime::{kv::CacheConfig, scheduler::SchedulerConfig};

use crate::MemorySnapshot;

#[derive(Debug, Clone)]
/// Configuration shared by model loading, scheduling, and backend execution.
pub struct RuntimeConfig {
    /// K/V cache block layout, capacity, and storage type.
    pub kv_cache: CacheConfig,
    /// Derive the K/V block count from model shape and available device memory.
    ///
    /// Set this to `false` when `kv_cache.block_count` is an explicit limit.
    pub automatic_kv_cache: bool,
    /// Decode batching and scheduling policy.
    pub scheduler: SchedulerConfig,
    /// Resource policy applied before vision preprocessing and execution.
    pub vision: VisionRuntimeConfig,
    /// Accelerator-memory headroom retained outside model residency.
    pub memory: MemoryRuntimeConfig,
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
            automatic_kv_cache: true,
            scheduler: SchedulerConfig::default(),
            vision: VisionRuntimeConfig::default(),
            memory: MemoryRuntimeConfig::default(),
            #[cfg(feature = "cuda")]
            cuda: CudaConfig::default(),
            #[cfg(feature = "metal")]
            metal: MetalConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Runtime policy for accelerator-memory safety headroom.
pub struct MemoryRuntimeConfig {
    /// Percentage of total accelerator memory retained as hard headroom.
    ///
    /// `None` uses the backend-aware default. An explicit zero disables the
    /// percentage reserve.
    pub reserve_percent: Option<u8>,
    /// Absolute hard-reserve floor in bytes.
    ///
    /// `None` uses the backend-aware default. When either override is set, the
    /// larger configured reserve is used instead of the default policy.
    pub reserve_bytes: Option<u64>,
}

impl MemoryRuntimeConfig {
    /// Resolves the hard reserve for one accelerator memory snapshot.
    #[must_use]
    pub fn hard_reserve_bytes(self, memory: &MemorySnapshot) -> u64 {
        if self.reserve_percent.is_some() || self.reserve_bytes.is_some() {
            let percent = self
                .reserve_percent
                .zip(memory.total_bytes)
                .map_or(0, |(percent, total)| percentage(total, percent));
            return percent.max(self.reserve_bytes.unwrap_or_default());
        }
        let divisor = if memory.unified {
            4
        } else {
            10
        };
        memory
            .total_bytes
            .map_or(GIB, |total| (total / divisor).max(GIB))
            .saturating_add(memory.allocation_reserve_bytes)
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

fn percentage(total: u64, percent: u8) -> u64 {
    let percent = u64::from(percent);
    (total / 100)
        .saturating_mul(percent)
        .saturating_add((total % 100).saturating_mul(percent) / 100)
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
