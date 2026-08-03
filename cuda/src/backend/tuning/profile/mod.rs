use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use mircuda::DeviceInfo;
use runtime::tuning::StartupBudget;

use self::storage::{DeviceKey, StoredProfile, cache_name, load, persist};
use super::{CudaTuningConfig, CudaTuningMode};
use crate::{
    AttentionExecution, AttentionPlanRequest, DenseExecution, DensePlanRequest, PlanSource,
};

mod attention;
mod dense;
mod moe;
mod quantized;
mod storage;
#[cfg(test)]
mod tests;

pub(in crate::backend) use moe::{
    AffineMoeExecution, ClampedMoeExecution, ClampedMoeStorage, MoeProfileExecution,
    MoeProfileRequest, MxFp4MoeExecution, MxFp4MoeStorage, MxFp8MoeExecution, MxFp8MoeStorage,
};
pub(in crate::backend) use quantized::{
    AffineProjectionExecution, DirectFp8ProjectionExecution, DirectFp8ScaleDType,
    DirectFp8WeightScale, MxFp8ProjectionExecution, NvFp4WeightOnlyExecution,
    QuantizedProfileExecution, QuantizedProfileRequest,
};

#[derive(Clone, Debug)]
pub struct CudaAutoTuner {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub(super) struct Inner {
    config: CudaTuningConfig,
    cache_path: Option<PathBuf>,
    device: DeviceKey,
    state: Mutex<State>,
}

#[derive(Debug)]
pub(super) struct State {
    pub(super) dense: HashMap<DensePlanRequest, DenseRuntimeEntry>,
    pub(super) dense_inflight: HashSet<DensePlanRequest>,
    pub(super) attention: HashMap<AttentionProfileRequest, AttentionRuntimeEntry>,
    pub(super) attention_inflight: HashSet<AttentionProfileRequest>,
    pub(super) moe: HashMap<MoeProfileRequest, MoeRuntimeEntry>,
    pub(super) moe_inflight: HashSet<MoeProfileRequest>,
    pub(super) quantized: HashMap<QuantizedProfileRequest, QuantizedRuntimeEntry>,
    pub(super) quantized_inflight: HashSet<QuantizedProfileRequest>,
    pub(super) budget: StartupBudget,
    pub(super) sealed: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DenseRuntimeEntry {
    pub(super) execution: DenseExecution,
    pub(super) source: PlanSource,
    pub(super) average_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionRuntimeEntry {
    pub(super) execution: AttentionExecution,
    pub(super) source: PlanSource,
    pub(super) average_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MoeRuntimeEntry {
    pub(super) execution: MoeProfileExecution,
    pub(super) source: PlanSource,
    pub(super) average_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct QuantizedRuntimeEntry {
    pub(super) execution: QuantizedProfileExecution,
    pub(super) source: PlanSource,
    pub(super) average_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum AttentionFamily {
    Paged,
    ClampedSink,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AttentionProfileRequest {
    pub(crate) family: AttentionFamily,
    pub(crate) plan: AttentionPlanRequest,
    pub(crate) block_size: usize,
    pub(crate) dtype: runtime::kv::KvCacheDType,
    pub(crate) window_tokens: Option<usize>,
}

impl CudaAutoTuner {
    pub(in crate::backend) fn new(device: &DeviceInfo, config: CudaTuningConfig) -> Self {
        let device = DeviceKey {
            name: device.name.clone(),
            compute_capability: device.compute_capability,
            multiprocessors: device.multiprocessor_count,
            integrated: device.integrated,
        };
        let cache_path = config
            .cache_directory
            .as_ref()
            .map(|directory| directory.join(cache_name(&device)));
        let stored = cache_path.as_deref().and_then(|path| load(path, &device)).unwrap_or_default();
        let dense = stored
            .dense
            .into_iter()
            .map(|entry| {
                (
                    entry.request,
                    DenseRuntimeEntry {
                        execution: entry.execution,
                        source: PlanSource::MeasuredCache,
                        average_ns: entry.average_ns,
                    },
                )
            })
            .collect();
        let attention = stored
            .attention
            .into_iter()
            .map(|entry| {
                (
                    entry.request,
                    AttentionRuntimeEntry {
                        execution: entry.execution,
                        source: PlanSource::MeasuredCache,
                        average_ns: entry.average_ns,
                    },
                )
            })
            .collect();
        let moe = stored
            .moe
            .into_iter()
            .map(|entry| {
                (
                    entry.request,
                    MoeRuntimeEntry {
                        execution: entry.execution,
                        source: PlanSource::MeasuredCache,
                        average_ns: entry.average_ns,
                    },
                )
            })
            .collect();
        let quantized = stored
            .quantized
            .into_iter()
            .map(|entry| {
                (
                    entry.request,
                    QuantizedRuntimeEntry {
                        execution: entry.execution,
                        source: PlanSource::MeasuredCache,
                        average_ns: entry.average_ns,
                    },
                )
            })
            .collect();
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    dense,
                    dense_inflight: HashSet::new(),
                    attention,
                    attention_inflight: HashSet::new(),
                    moe,
                    moe_inflight: HashSet::new(),
                    quantized,
                    quantized_inflight: HashSet::new(),
                    budget: StartupBudget::new(std::time::Duration::from_millis(
                        config.startup_budget_ms,
                    )),
                    sealed: false,
                }),
                config,
                cache_path,
                device,
            }),
        }
    }

    pub(crate) fn prepares_candidates(&self, source: PlanSource) -> bool {
        self.inner.config.mode == CudaTuningMode::Startup
            && source != PlanSource::ExplicitPolicy
            && self.inner.config.startup_budget_ms > 0
            && self.inner.state.lock().is_ok_and(|state| !state.sealed)
    }

    pub(crate) fn finish_startup(&self) {
        let Ok(mut state) = self.inner.state.lock() else {
            return;
        };
        state.sealed = true;
        state.dense_inflight.clear();
        state.attention_inflight.clear();
        state.moe_inflight.clear();
        state.quantized_inflight.clear();
    }

    pub(crate) fn iterations(&self, tokens: usize) -> (u32, u32) {
        let measured = match tokens {
            0..=255 => self.inner.config.measurement_iterations,
            256..=1_023 => self.inner.config.measurement_iterations.min(2),
            _ => 1,
        };
        (self.inner.config.warmup_iterations, measured.max(1))
    }

    pub(crate) fn minimum_improvement_bps(&self) -> u16 {
        self.inner.config.minimum_improvement_bps
    }

    fn persist(&self, profile: StoredProfile) {
        let Some(path) = &self.inner.cache_path else {
            return;
        };
        if let Err(error) = persist(path, &self.inner.device, profile) {
            tracing::warn!(path = %path.display(), %error, "failed to persist CUDA tuning profile");
        }
    }

    fn snapshot(state: &State) -> StoredProfile {
        StoredProfile {
            dense: dense::stored_entries(&state.dense),
            attention: attention::stored_entries(&state.attention),
            moe: storage::stored_moe_entries(&state.moe),
            quantized: quantized::stored_entries(&state.quantized),
        }
    }
}
