use runtime::Result as RuntimeResult;

use super::{Engine, EngineInner};
use crate::MemorySnapshot;

impl Engine {
    pub(crate) fn memory_snapshot(&self) -> RuntimeResult<MemorySnapshot> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let memory = cuda.memory_stats()?;
                Ok(MemorySnapshot {
                    total_bytes: Some(memory.total),
                    available_bytes: Some(memory.available),
                    active_bytes: memory.used,
                    cached_bytes: memory.reserved.saturating_sub(memory.used),
                    source: format!("CUDA device memory — {}", memory.device),
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
                    source: "Metal unified memory".to_owned(),
                    unified: true,
                })
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => super::unavailable(),
        }
    }
}
