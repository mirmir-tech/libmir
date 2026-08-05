use std::time::Duration;

use super::tuning::Execution;
use crate::{
    PlanSource,
    backend::tuning::{NvFp4WeightOnlyExecution, QuantizedProfileRequest},
};

impl From<NvFp4WeightOnlyExecution> for Execution {
    fn from(value: NvFp4WeightOnlyExecution) -> Self {
        match value {
            NvFp4WeightOnlyExecution::Compressed => Self::Compressed,
            NvFp4WeightOnlyExecution::TensorCore => Self::TensorCore,
            NvFp4WeightOnlyExecution::MarlinN128K128 => Self::MarlinN128K128,
            NvFp4WeightOnlyExecution::MarlinN128K64 => Self::MarlinN128K64,
            NvFp4WeightOnlyExecution::MarlinN64K128 => Self::MarlinN64K128,
            NvFp4WeightOnlyExecution::Materialized => Self::Materialized,
        }
    }
}

impl From<Execution> for NvFp4WeightOnlyExecution {
    fn from(value: Execution) -> Self {
        match value {
            Execution::Compressed => Self::Compressed,
            Execution::TensorCore => Self::TensorCore,
            Execution::MarlinN128K128 => Self::MarlinN128K128,
            Execution::MarlinN128K64 => Self::MarlinN128K64,
            Execution::MarlinN64K128 => Self::MarlinN64K128,
            Execution::Materialized => Self::Materialized,
        }
    }
}

pub(super) fn trace(
    request: QuantizedProfileRequest,
    execution: Execution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?request,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA NVFP4 W4A16 projection execution"
    );
}
