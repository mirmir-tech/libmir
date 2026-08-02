use std::time::Duration;

use mircuda::{
    Context, CublasLtBf16Plan, CublasLtBf16Spec, DenseMatmulPlan, DenseMatmulSpec, DenseVectorPlan,
    DenseVectorSpec, DeviceBuffer, Stream, bf16,
};

use super::super::CudaBackend;
use crate::{DenseExecution, DensePlanRequest, Error, Result};

#[derive(Debug)]
pub(super) struct Candidate {
    pub(super) execution: DenseExecution,
    pub(super) plan: Plan,
}

#[derive(Debug)]
pub(super) enum Plan {
    Matrix(DenseMatmulPlan<bf16>),
    Vector(DenseVectorPlan<bf16>),
    Vendor(CublasLtBf16Plan),
}

impl Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        request: DensePlanRequest,
        execution: DenseExecution,
    ) -> Result<Self> {
        Self::new_with_resources(&backend.inner.context, &backend.inner.stream, request, execution)
    }

    pub(super) fn new_with_resources(
        context: &Context,
        stream: &Stream,
        request: DensePlanRequest,
        execution: DenseExecution,
    ) -> Result<Self> {
        let plan = match execution {
            DenseExecution::Matrix => Plan::Matrix(DenseMatmulPlan::new(
                context,
                stream,
                DenseMatmulSpec::new(
                    request.tokens,
                    request.output_features,
                    request.input_features,
                )?,
            )?),
            DenseExecution::Vector if request.tokens == 1 => Plan::Vector(DenseVectorPlan::new(
                context,
                stream,
                DenseVectorSpec::new(request.output_features, request.input_features)?,
            )?),
            DenseExecution::CublasLt => Plan::Vendor(CublasLtBf16Plan::new(
                context,
                stream,
                CublasLtBf16Spec::new(
                    request.tokens,
                    request.output_features,
                    request.input_features,
                )?,
            )?),
            DenseExecution::Vector => {
                return Err(Error::InvalidExecutionPlan("BF16 vector tuning requires one token"));
            },
            DenseExecution::BlockFp8Vector | DenseExecution::Fp8Int4Vector => {
                return Err(Error::InvalidExecutionPlan(
                    "BF16 tuner received a compressed-weight execution",
                ));
            },
        };
        Ok(Self { execution, plan })
    }
}

impl Plan {
    pub(super) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Matrix(plan) => Ok(plan.execute(stream, input, weight, output, 1.0, 0.0)?),
            Self::Vector(plan) => Ok(plan.execute(stream, input, weight, output, 1.0, 0.0)?),
            Self::Vendor(plan) => Ok(plan.execute(stream, input, weight, output, 1.0, 0.0)?),
        }
    }
}

pub(super) fn candidates(request: DensePlanRequest) -> Vec<DenseExecution> {
    let mut executions = vec![DenseExecution::Matrix, DenseExecution::CublasLt];
    if request.tokens == 1 {
        executions.push(DenseExecution::Vector);
    }
    executions
}

pub(super) fn initial_executions(
    planned: DenseExecution,
    cached: Option<DenseExecution>,
    phase: crate::ExecutionPhase,
) -> Vec<DenseExecution> {
    if let Some(cached) = cached {
        return if cached == planned {
            vec![cached]
        } else {
            vec![cached, planned]
        };
    }
    if phase == crate::ExecutionPhase::Prefill && planned != DenseExecution::CublasLt {
        vec![DenseExecution::CublasLt, planned]
    } else {
        vec![planned]
    }
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub(super) fn measure(
    context: &Context,
    stream: &Stream,
    plan: &mut Plan,
    input: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
    iterations: u32,
) -> Result<Duration> {
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..iterations {
        plan.execute(stream, input, weight, output)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(Duration::from_secs_f32(
        started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionPhase;

    #[test]
    fn prefill_uses_vendor_then_planner_fallback() {
        assert_eq!(
            initial_executions(DenseExecution::Matrix, None, ExecutionPhase::Prefill),
            [DenseExecution::CublasLt, DenseExecution::Matrix]
        );
        assert_eq!(
            initial_executions(DenseExecution::Vector, None, ExecutionPhase::Decode),
            [DenseExecution::Vector]
        );
    }

    #[test]
    fn cached_execution_precedes_planner_fallback() {
        assert_eq!(
            initial_executions(
                DenseExecution::Matrix,
                Some(DenseExecution::Vector),
                ExecutionPhase::Decode
            ),
            [DenseExecution::Vector, DenseExecution::Matrix]
        );
    }
}
