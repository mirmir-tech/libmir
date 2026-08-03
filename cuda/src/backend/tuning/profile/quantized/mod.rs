use std::time::Duration;

use super::{CudaAutoTuner, QuantizedRuntimeEntry};
use crate::{ExecutionPhase, PlanSource};

mod entries;
pub(super) use entries::stored_entries;

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
    TensorCore,
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
    Materialized,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) enum QuantizedProfileExecution {
    Affine(AffineProjectionExecution),
    MxFp8(MxFp8ProjectionExecution),
    DirectFp8(DirectFp8ProjectionExecution),
    NvFp4WeightOnly(NvFp4WeightOnlyExecution),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
enum QuantizedProfileFormat {
    Affine {
        group_size: usize,
        bits: usize,
    },
    MxFp8,
    DirectFp8DynamicE4M3OutputChannel {
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    },
    DirectFp8StaticE4M3 {
        weight_scale: DirectFp8WeightScale,
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    },
    DirectFp8Bf16E5M2WeightOnly {
        bias: bool,
    },
    NvFp4Bf16WeightOnly,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) struct QuantizedProfileRequest {
    phase: ExecutionPhase,
    tokens: usize,
    input_features: usize,
    output_features: usize,
    format: QuantizedProfileFormat,
}

impl QuantizedProfileRequest {
    #[must_use]
    pub(in crate::backend) const fn tokens(self) -> usize {
        self.tokens
    }

    pub(in crate::backend) const fn affine(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        group_size: usize,
        bits: usize,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format: QuantizedProfileFormat::Affine { group_size, bits },
        }
    }

    pub(in crate::backend) const fn mxfp8(
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format: QuantizedProfileFormat::MxFp8,
        }
    }

    pub(in crate::backend) const fn direct_fp8_dynamic_e4m3(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format: QuantizedProfileFormat::DirectFp8DynamicE4M3OutputChannel { scale_dtype, bias },
        }
    }

    pub(in crate::backend) const fn direct_fp8_static_e4m3(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        weight_scale: DirectFp8WeightScale,
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format: QuantizedProfileFormat::DirectFp8StaticE4M3 { weight_scale, scale_dtype, bias },
        }
    }

    pub(in crate::backend) const fn direct_fp8_bf16_e5m2_weight_only(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        bias: bool,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format: QuantizedProfileFormat::DirectFp8Bf16E5M2WeightOnly { bias },
        }
    }

    pub(in crate::backend) const fn nvfp4_bf16_weight_only(
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format: QuantizedProfileFormat::NvFp4Bf16WeightOnly,
        }
    }
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
