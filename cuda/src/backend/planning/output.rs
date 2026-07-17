use super::{
    CudaExecutionPlanner, CudaKernelAdmission, CudaNumericalPolicy, CudaOutputHeadPolicy,
    PlanSource,
};
use crate::{Error, Result};

/// Prepared output-projection implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OutputHeadExecution {
    AutoRefined,
    Bf16,
    Fp8Blockwise,
    Fp8Vectorized,
    Fp8Residual,
    Fp8BlockVectorized,
    Fp8BlockRefined,
}

/// Geometry used to select output-head storage and execution.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OutputHeadPlanRequest {
    pub input_features: usize,
    pub output_features: usize,
}

/// Immutable output-head decision made before session construction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OutputHeadPlan {
    execution: OutputHeadExecution,
    source: PlanSource,
}

impl OutputHeadPlan {
    #[must_use]
    pub const fn execution(self) -> OutputHeadExecution {
        self.execution
    }

    #[must_use]
    pub const fn source(self) -> PlanSource {
        self.source
    }
}

impl CudaExecutionPlanner {
    pub fn plan_output_head(&self, request: OutputHeadPlanRequest) -> Result<OutputHeadPlan> {
        if request.input_features == 0 || request.output_features == 0 {
            return Err(Error::InvalidExecutionPlan("output-head dimensions are empty"));
        }
        let policy = self.policy();
        if policy.output_head == CudaOutputHeadPolicy::Auto
            && self.hardware().compute_capability().0 == 12
            && tuned_refinement(request)
        {
            return Ok(OutputHeadPlan {
                execution: OutputHeadExecution::AutoRefined,
                source: PlanSource::Tuned,
            });
        }
        let fp8 = policy.numerical == CudaNumericalPolicy::Throughput
            && policy.admission == CudaKernelAdmission::Experimental
            && self.hardware().compute_capability().0 >= 12;
        Ok(
            if fp8
                && !matches!(
                    policy.output_head,
                    CudaOutputHeadPolicy::Auto | CudaOutputHeadPolicy::Bf16
                )
            {
                let execution = match policy.output_head {
                    CudaOutputHeadPolicy::Fp8Blockwise => OutputHeadExecution::Fp8Blockwise,
                    CudaOutputHeadPolicy::Fp8Vectorized => OutputHeadExecution::Fp8Vectorized,
                    CudaOutputHeadPolicy::Fp8Residual => OutputHeadExecution::Fp8Residual,
                    CudaOutputHeadPolicy::Fp8BlockVectorized => {
                        OutputHeadExecution::Fp8BlockVectorized
                    },
                    CudaOutputHeadPolicy::Fp8BlockRefined => OutputHeadExecution::Fp8BlockRefined,
                    CudaOutputHeadPolicy::Auto | CudaOutputHeadPolicy::Bf16 => unreachable!(),
                };
                OutputHeadPlan {
                    execution,
                    source: PlanSource::ExplicitPolicy,
                }
            } else {
                OutputHeadPlan {
                    execution: OutputHeadExecution::Bf16,
                    source: PlanSource::Fallback,
                }
            },
        )
    }
}

const fn tuned_refinement(request: OutputHeadPlanRequest) -> bool {
    matches!(
        (request.input_features, request.output_features),
        (2_816, 262_144) | (4_096, 151_936)
    )
}
