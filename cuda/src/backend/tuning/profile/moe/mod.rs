use std::time::Duration;

use super::{CudaAutoTuner, MoeRuntimeEntry};
use crate::{ExecutionPhase, GatedActivation, MoeExecution, PlanSource};

mod execution;
mod request;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum AffineMoeExecution {
    FusedGated,
    SeparatePair,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum ClampedMoeExecution {
    FusedReduce,
    RouteParallel,
    MarlinN128K128,
    MarlinN128K64,
    MarlinN64K128,
    MarlinM64N256K64,
    MarlinM64N128K64,
    MarlinM64N64K128,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum ClampedMoeStorage {
    Native,
    Mlx,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum MxFp4MoeExecution {
    SingleWarp,
    EightWarps,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum MxFp4MoeStorage {
    Separate,
    Interleaved,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum MxFp8MoeExecution {
    FourWarps,
    EightWarps,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum MxFp8MoeStorage {
    Separate,
    Interleaved,
}

impl MxFp4MoeExecution {
    pub(in crate::backend) const fn warps_per_block(self) -> usize {
        match self {
            Self::SingleWarp => 1,
            Self::EightWarps => 8,
        }
    }
}

impl MxFp8MoeExecution {
    pub(in crate::backend) const fn warps_per_block(self) -> usize {
        match self {
            Self::FourWarps => 4,
            Self::EightWarps => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum MoeProfileExecution {
    NvFp4(MoeExecution),
    Affine(AffineMoeExecution),
    Clamped(ClampedMoeExecution),
    MxFp4(MxFp4MoeExecution),
    MxFp8(MxFp8MoeExecution),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(super) enum MoeProfileFormat {
    NvFp4 {
        activation: GatedActivation,
        weight_only: bool,
    },
    Affine {
        group_size: usize,
        bits: usize,
        activation: GatedActivation,
    },
    Clamped {
        storage: ClampedMoeStorage,
    },
    MxFp4 {
        storage: MxFp4MoeStorage,
        activation: GatedActivation,
    },
    MxFp8 {
        storage: MxFp8MoeStorage,
        bias: bool,
        activation: GatedActivation,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) struct MoeProfileRequest {
    pub(super) phase: ExecutionPhase,
    pub(super) tokens: usize,
    pub(super) experts: usize,
    pub(super) top_k: usize,
    pub(super) hidden_features: usize,
    pub(super) intermediate_features: usize,
    pub(super) format: MoeProfileFormat,
}

impl CudaAutoTuner {
    pub(in crate::backend) fn prepares_moe_candidates(
        &self,
        request: MoeProfileRequest,
        source: PlanSource,
    ) -> bool {
        self.inner.config.mode == super::CudaTuningMode::Startup
            && source != PlanSource::ExplicitPolicy
            && self.inner.config.startup_budget_ms > 0
            && self
                .inner
                .state
                .lock()
                .is_ok_and(|state| !state.sealed || request.late_decode_allowed())
    }

    pub(in crate::backend) fn lookup_moe(
        &self,
        request: MoeProfileRequest,
    ) -> Option<(MoeProfileExecution, PlanSource)> {
        if self.inner.config.mode == super::CudaTuningMode::Disabled {
            return None;
        }
        self.inner
            .state
            .lock()
            .ok()?
            .moe
            .get(&request)
            .map(|entry| (entry.execution, entry.source))
    }

    pub(in crate::backend) fn lookup_clamped_moe(
        &self,
        request: MoeProfileRequest,
    ) -> Option<(MoeProfileExecution, PlanSource)> {
        if self.inner.config.mode == super::CudaTuningMode::Disabled {
            return None;
        }
        let state = self.inner.state.lock().ok()?;
        if let Some(entry) = state.moe.get(&request) {
            return Some((entry.execution, entry.source));
        }
        state
            .moe
            .iter()
            .filter(|(candidate, entry)| {
                request.same_clamped_geometry(**candidate)
                    && matches!(
                        entry.execution,
                        MoeProfileExecution::Clamped(
                            ClampedMoeExecution::MarlinN128K128
                                | ClampedMoeExecution::MarlinN128K64
                                | ClampedMoeExecution::MarlinN64K128
                                | ClampedMoeExecution::MarlinM64N256K64
                                | ClampedMoeExecution::MarlinM64N128K64
                                | ClampedMoeExecution::MarlinM64N64K128
                        )
                    )
            })
            .min_by_key(|(candidate, _)| candidate.tokens.abs_diff(request.tokens))
            .map(|(_, entry)| (entry.execution, entry.source))
    }

    pub(in crate::backend) fn claim_moe(&self, request: MoeProfileRequest) -> bool {
        let Ok(mut state) = self.inner.state.lock() else {
            return false;
        };
        self.inner.config.mode == super::CudaTuningMode::Startup
            && (!state.sealed || request.late_decode_allowed())
            && state.budget.available()
            && !state.moe.contains_key(&request)
            && state.moe_inflight.insert(request)
    }

    pub(in crate::backend) fn record_moe(
        &self,
        request: MoeProfileRequest,
        execution: MoeProfileExecution,
        average: Duration,
        tuning_elapsed: Duration,
    ) {
        let snapshot = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            state.moe_inflight.remove(&request);
            state.budget.consume(tuning_elapsed);
            state.moe.insert(
                request,
                MoeRuntimeEntry {
                    execution,
                    source: PlanSource::MeasuredStartup,
                    average_ns: u64::try_from(average.as_nanos()).unwrap_or(u64::MAX),
                },
            );
            Self::snapshot(&state)
        };
        self.persist(snapshot);
    }

    pub(in crate::backend) fn abandon_moe(&self, request: MoeProfileRequest) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.moe_inflight.remove(&request);
        }
    }
}

impl MoeProfileRequest {
    pub(super) const fn late_decode_allowed(self) -> bool {
        matches!(self.phase, ExecutionPhase::Decode)
            && matches!(self.format, MoeProfileFormat::NvFp4 { weight_only: true, .. })
    }

    fn same_clamped_geometry(self, other: Self) -> bool {
        self.experts == other.experts
            && self.top_k == other.top_k
            && self.hidden_features == other.hidden_features
            && self.intermediate_features == other.intermediate_features
            && matches!(
                (self.format, other.format),
                (
                    MoeProfileFormat::Clamped { storage: left },
                    MoeProfileFormat::Clamped { storage: right }
                ) if left == right
            )
    }
}
