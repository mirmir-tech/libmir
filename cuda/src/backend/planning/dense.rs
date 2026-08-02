use super::{CudaExecutionPlanner, ExecutionPhase, PlanSource};
use crate::{Error, Result};

/// Semantic role of a dense projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum DenseRole {
    AttentionQkv,
    AttentionOutput,
    DenseGateUp,
    DenseDown,
    Router,
    OutputHead,
}

/// Model-level dense implementation selected for a fixed shape.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum DenseExecution {
    Matrix,
    Vector,
    CublasLt,
    BlockFp8Vector,
    Fp8Int4Vector,
}

/// Complete generic key for a dense plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct DensePlanRequest {
    pub phase: ExecutionPhase,
    pub role: DenseRole,
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
}

/// Selected dense implementation and its provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DensePlan {
    execution: DenseExecution,
    source: PlanSource,
}

impl DensePlan {
    #[must_use]
    pub const fn execution(self) -> DenseExecution {
        self.execution
    }

    #[must_use]
    pub const fn source(self) -> PlanSource {
        self.source
    }
}

impl CudaExecutionPlanner {
    pub fn plan_dense(self, request: DensePlanRequest) -> Result<DensePlan> {
        self.plan_dense_with_weights(request, false)
    }

    pub(in crate::backend) fn plan_dense_with_prepared_weights(
        self,
        request: DensePlanRequest,
    ) -> Result<DensePlan> {
        self.plan_dense_with_weights(request, true)
    }

    fn plan_dense_with_weights(
        self,
        request: DensePlanRequest,
        compressed_weights_available: bool,
    ) -> Result<DensePlan> {
        validate(request)?;
        let sm12 = self.hardware().compute_capability().0 == 12;
        let output_head = request.role == DenseRole::OutputHead;
        let policy = self.policy();
        let quantized = compressed_weights_available
            && policy.numerical == super::CudaNumericalPolicy::Throughput
            && policy.admission == super::CudaKernelAdmission::Experimental
            && matches!(
                request.role,
                DenseRole::AttentionOutput | DenseRole::DenseGateUp | DenseRole::DenseDown
            )
            && request.phase == ExecutionPhase::Decode
            && request.tokens == 1
            && request.input_features.is_multiple_of(128)
            && request.output_features.is_multiple_of(128);
        let block_fp8 = quantized
            && policy.dense_weights == super::CudaDenseWeightPolicy::BlockFp8Role(request.role);
        let fp8_int4 = quantized
            && policy.dense_weights == super::CudaDenseWeightPolicy::Fp8Int4Role(request.role);
        let selected = match policy.dense_vectors {
            super::CudaDenseVectorPolicy::Disabled => false,
            super::CudaDenseVectorPolicy::Tuned => true,
            super::CudaDenseVectorPolicy::Role(role) => role == request.role,
        };
        let vendor_selected = match policy.dense_vendor {
            super::CudaDenseVendorPolicy::Disabled => false,
            super::CudaDenseVendorPolicy::Tuned => true,
            super::CudaDenseVendorPolicy::Role(role) => role == request.role,
        };
        let explicit_vendor = policy.numerical == super::CudaNumericalPolicy::Throughput
            && policy.admission == super::CudaKernelAdmission::Experimental
            && vendor_selected
            && request.input_features.is_multiple_of(8)
            && request.output_features.is_multiple_of(8);
        let vendor = explicit_vendor;
        let experimental = policy.numerical == super::CudaNumericalPolicy::Throughput
            && policy.admission == super::CudaKernelAdmission::Experimental
            && selected
            && request.phase == ExecutionPhase::Decode
            && request.tokens == 1
            && request.input_features.is_multiple_of(2);
        let tuned_vector = sm12 && tuned_decode_attention_vector(request);
        let vector = sm12
            && request.phase == ExecutionPhase::Decode
            && request.tokens == 1
            && (output_head || tuned_vector || experimental);
        let plan = DensePlan {
            execution: if fp8_int4 {
                DenseExecution::Fp8Int4Vector
            } else if block_fp8 {
                DenseExecution::BlockFp8Vector
            } else if vendor {
                DenseExecution::CublasLt
            } else if vector {
                DenseExecution::Vector
            } else {
                DenseExecution::Matrix
            },
            source: if block_fp8 || fp8_int4 || experimental || explicit_vendor {
                PlanSource::ExplicitPolicy
            } else if vector {
                PlanSource::Heuristic
            } else {
                PlanSource::Fallback
            },
        };
        let hardware = self.hardware();
        tracing::debug!(
            target: "libmir::cuda::planning",
            compute_major = hardware.compute_capability().0,
            compute_minor = hardware.compute_capability().1,
            multiprocessors = hardware.multiprocessor_count().get(),
            total_memory = hardware.total_memory().get(),
            memory_architecture = ?hardware.memory_architecture(),
            numerical_policy = ?policy.numerical,
            kernel_admission = ?policy.admission,
            dense_vector_policy = ?policy.dense_vectors,
            dense_vendor_policy = ?policy.dense_vendor,
            dense_weight_policy = ?policy.dense_weights,
            compressed_weights_available,
            phase = ?request.phase,
            role = ?request.role,
            tokens = request.tokens,
            input_features = request.input_features,
            output_features = request.output_features,
            execution = ?plan.execution,
            source = ?plan.source,
            "selected CUDA dense plan"
        );
        Ok(plan)
    }
}

const fn tuned_decode_attention_vector(request: DensePlanRequest) -> bool {
    matches!(request.phase, ExecutionPhase::Decode)
        && request.tokens == 1
        && request.input_features.is_multiple_of(2)
        && (matches!(
            request.role,
            DenseRole::AttentionQkv | DenseRole::AttentionOutput | DenseRole::Router
        ) || (matches!(request.role, DenseRole::DenseDown)
            && request.input_features / request.output_features >= 2))
}

fn validate(request: DensePlanRequest) -> Result<()> {
    if request.tokens == 0 || request.input_features == 0 || request.output_features == 0 {
        Err(Error::InvalidExecutionPlan("dense plan has an empty dimension"))
    } else {
        Ok(())
    }
}
