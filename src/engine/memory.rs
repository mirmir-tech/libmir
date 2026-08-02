use runtime::Result as RuntimeResult;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use sysinfo::System;

use super::{Engine, EngineInner};
use crate::MemorySnapshot;

#[cfg(feature = "cuda")]
const CUDA_UNIFIED_ALLOCATION_RESERVE: u64 = 4 * 1024 * 1024 * 1024;

impl Engine {
    pub(crate) fn memory_snapshot(&self) -> RuntimeResult<MemorySnapshot> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let memory = cuda.memory_stats()?;
                let (total, available, source) = cuda_memory(&memory);
                Ok(MemorySnapshot {
                    total_bytes: Some(total),
                    available_bytes: Some(available),
                    active_bytes: memory.used,
                    cached_bytes: memory.reserved.saturating_sub(memory.used),
                    allocation_reserve_bytes: if memory.integrated {
                        CUDA_UNIFIED_ALLOCATION_RESERVE
                    } else {
                        0
                    },
                    source,
                    unified: memory.integrated,
                })
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => {
                let memory = metal.memory_stats()?;
                let total = memory.recommended.or(Some(memory.limit)).filter(|value| *value > 0);
                let retained = memory.active.saturating_add(memory.cached);
                Ok(MemorySnapshot {
                    total_bytes: total,
                    available_bytes: total.map(|value| value.saturating_sub(retained)),
                    active_bytes: memory.active,
                    cached_bytes: memory.cached,
                    allocation_reserve_bytes: 0,
                    source: "Metal unified memory".to_owned(),
                    unified: true,
                })
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_memory(memory: &cuda::CudaMemoryStats) -> (u64, u64, String) {
    #[cfg(target_os = "linux")]
    if memory.integrated {
        let mut system = System::new();
        system.refresh_memory();
        let host_total = system.total_memory();
        let host_available = system.available_memory();
        if host_total > 0 {
            let (total, available) = unified_cuda_limits(memory, host_total, host_available);
            return (
                total,
                available,
                format!("CUDA unified memory bounded by host — {}", memory.device),
            );
        }
    }
    (
        memory.total,
        memory.available,
        format!("CUDA device memory — {}", memory.device),
    )
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn unified_cuda_limits(
    memory: &cuda::CudaMemoryStats,
    host_total: u64,
    host_available: u64,
) -> (u64, u64) {
    let total = memory.total.min(host_total);
    // On integrated CUDA devices `cuMemGetInfo` reports immediately free
    // pages. Linux `MemAvailable` also includes reclaimable page cache, which
    // is the usable capacity of the shared host/device memory pool.
    (total, host_available.min(total))
}

#[cfg(all(test, feature = "cuda", target_os = "linux"))]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn unified_cuda_uses_reclaimable_host_capacity() {
        let memory = cuda::CudaMemoryStats {
            total: 128 * GIB,
            available: 40 * GIB,
            reserved: 0,
            used: 0,
            device: "integrated".to_owned(),
            integrated: true,
        };

        assert_eq!(unified_cuda_limits(&memory, 120 * GIB, 110 * GIB), (120 * GIB, 110 * GIB));
    }
}
