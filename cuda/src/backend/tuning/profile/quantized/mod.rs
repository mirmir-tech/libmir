use std::time::Duration;

use super::{CudaAutoTuner, QuantizedRuntimeEntry};
use crate::PlanSource;

mod entries;
mod request;
pub(super) use entries::stored_entries;
pub(in crate::backend) use request::QuantizedProfileRequest;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum AffineProjectionExecution {
    Qmm,
    Gemv,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum MxFp8ProjectionExecution {
    Portable,
    TensorCore,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum DirectFp8ProjectionExecution {
    Portable,
    PortableCached,
    TensorCore,
    TensorCoreWide,
    CublasLt,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum DirectFp8ScaleDType {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum DirectFp8WeightScale {
    Tensor,
    OutputChannel,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum NvFp4WeightOnlyExecution {
    Compressed,
    TensorCore,
    MarlinN128K128,
    MarlinN128K64,
    MarlinN64K128,
    Materialized,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum QuantizedProfileExecution {
    Affine(AffineProjectionExecution),
    MxFp8(MxFp8ProjectionExecution),
    DirectFp8(DirectFp8ProjectionExecution),
    NvFp4WeightOnly(NvFp4WeightOnlyExecution),
}

impl CudaAutoTuner {
    pub(in crate::backend) fn lookup_quantized(
        &self,
        request: QuantizedProfileRequest,
    ) -> Option<(QuantizedProfileExecution, PlanSource)> {
        if self.inner.config.mode == super::CudaTuningMode::Disabled {
            return None;
        }
        self.inner
            .state
            .lock()
            .ok()?
            .quantized
            .get(&request)
            .map(|entry| (entry.execution, entry.source))
    }

    pub(in crate::backend) fn claim_quantized(&self, request: QuantizedProfileRequest) -> bool {
        let Ok(mut state) = self.inner.state.lock() else {
            return false;
        };
        self.inner.config.mode == super::CudaTuningMode::Startup
            && !state.sealed
            && state.budget.available()
            && !state.quantized.contains_key(&request)
            && state.quantized_inflight.insert(request)
    }

    pub(in crate::backend) fn record_quantized(
        &self,
        request: QuantizedProfileRequest,
        execution: QuantizedProfileExecution,
        average: Duration,
        tuning_elapsed: Duration,
    ) {
        let snapshot = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            state.quantized_inflight.remove(&request);
            state.budget.consume(tuning_elapsed);
            state.quantized.insert(
                request,
                QuantizedRuntimeEntry {
                    execution,
                    source: PlanSource::MeasuredStartup,
                    average_ns: u64::try_from(average.as_nanos()).unwrap_or(u64::MAX),
                },
            );
            Self::snapshot(&state)
        };
        self.persist(snapshot);
    }

    pub(in crate::backend) fn abandon_quantized(&self, request: QuantizedProfileRequest) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.quantized_inflight.remove(&request);
        }
    }
}
