use super::{CudaExecutionPlanner, ExecutionPhase, PlanSource};
use crate::{Error, Result};

/// Quantized expert representation consumed by a planned `MoE` operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MoeQuantization {
    NvFp4,
}

/// Model-level routed expert implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MoeExecution {
    DirectW4A4,
    HybridW4A4,
    IndexedGrouped,
    FusedIndexedGrouped,
    SelectedWeightOnly,
    Bucketed,
}

/// Generic routed-expert selection key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MoePlanRequest {
    pub phase: ExecutionPhase,
    pub quantization: MoeQuantization,
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
    pub hidden_features: usize,
    pub intermediate_features: usize,
}

impl MoePlanRequest {
    #[must_use]
    pub const fn nvfp4(
        phase: ExecutionPhase,
        tokens: usize,
        experts: usize,
        top_k: usize,
        hidden_features: usize,
        intermediate_features: usize,
    ) -> Self {
        Self {
            phase,
            quantization: MoeQuantization::NvFp4,
            tokens,
            experts,
            top_k,
            hidden_features,
            intermediate_features,
        }
    }
}

/// Selected routed-expert implementation and its provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MoePlan {
    execution: MoeExecution,
    source: PlanSource,
}

impl MoePlan {
    #[must_use]
    pub const fn execution(self) -> MoeExecution {
        self.execution
    }

    #[must_use]
    pub const fn source(self) -> PlanSource {
        self.source
    }
}

impl CudaExecutionPlanner {
    pub fn plan_moe(self, request: MoePlanRequest) -> Result<MoePlan> {
        validate(request)?;
        let policy = self.policy();
        let weight_only = request.phase == ExecutionPhase::Decode
            && request.tokens > 1
            && policy.moe_batch == super::CudaMoeBatchPolicy::W4A16;
        let forced_bucketed = request.phase == ExecutionPhase::Decode
            && request.tokens > 1
            && policy.moe_batch == super::CudaMoeBatchPolicy::W4A4Bucketed;
        let routed_pairs = request.tokens.saturating_mul(request.top_k);
        let tuned_bucketed = request.phase == ExecutionPhase::Decode
            && request.tokens > 1
            && self.hardware().compute_capability().0 == 12
            && policy.moe_batch == super::CudaMoeBatchPolicy::Auto
            && routed_pairs >= request.experts.div_ceil(2);
        let fused = request.phase == ExecutionPhase::Decode
            && self.hardware().compute_capability().0 == 12
            && policy.numerical == super::CudaNumericalPolicy::Throughput
            && policy.admission == super::CudaKernelAdmission::Experimental
            && policy.moe_fusion == super::CudaMoeFusionPolicy::Tuned;
        let direct = request.phase == ExecutionPhase::Decode
            && self.hardware().compute_capability().0 == 12
            && policy.moe_batch == super::CudaMoeBatchPolicy::W4A4Direct
            && routed_pairs <= 20;
        let hybrid = request.phase == ExecutionPhase::Decode
            && self.hardware().compute_capability().0 == 12
            && policy.moe_batch == super::CudaMoeBatchPolicy::W4A4Hybrid
            && routed_pairs <= 20;
        let tuned_hybrid = request.phase == ExecutionPhase::Decode
            && request.tokens == 1
            && self.hardware().compute_capability().0 == 12
            && policy.moe_batch == super::CudaMoeBatchPolicy::Auto
            && routed_pairs <= 20;
        let execution = match request.phase {
            ExecutionPhase::Decode if weight_only => MoeExecution::SelectedWeightOnly,
            ExecutionPhase::Decode if forced_bucketed => MoeExecution::Bucketed,
            ExecutionPhase::Decode if fused => MoeExecution::FusedIndexedGrouped,
            ExecutionPhase::Decode if direct => MoeExecution::DirectW4A4,
            ExecutionPhase::Decode if hybrid => MoeExecution::HybridW4A4,
            ExecutionPhase::Decode if tuned_hybrid => MoeExecution::HybridW4A4,
            ExecutionPhase::Decode if tuned_bucketed => MoeExecution::Bucketed,
            ExecutionPhase::Decode => MoeExecution::IndexedGrouped,
            ExecutionPhase::Prefill => MoeExecution::Bucketed,
        };
        let source = if fused || direct || hybrid || weight_only || forced_bucketed {
            PlanSource::ExplicitPolicy
        } else if tuned_hybrid || self.hardware().compute_capability().0 == 12 {
            PlanSource::Tuned
        } else {
            PlanSource::Fallback
        };
        let plan = MoePlan { execution, source };
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
            moe_fusion_policy = ?policy.moe_fusion,
            phase = ?request.phase,
            quantization = ?request.quantization,
            tokens = request.tokens,
            experts = request.experts,
            top_k = request.top_k,
            routed_pairs,
            hidden_features = request.hidden_features,
            intermediate_features = request.intermediate_features,
            execution = ?plan.execution,
            source = ?plan.source,
            "selected CUDA MoE plan"
        );
        Ok(plan)
    }
}

fn validate(request: MoePlanRequest) -> Result<()> {
    let valid = request.tokens > 0
        && request.experts > 0
        && request.top_k > 0
        && request.top_k <= request.experts
        && request.hidden_features > 0
        && request.intermediate_features > 0;
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidExecutionPlan("MoE plan has invalid geometry"))
    }
}
