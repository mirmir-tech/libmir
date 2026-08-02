use std::num::{NonZeroU32, NonZeroUsize};

use mircuda::DeviceInfo;

use crate::{Error, Result};

mod attention;
mod dense;
mod moe;
mod output;
#[cfg(test)]
mod tests;

pub use dense::{DenseExecution, DensePlan, DensePlanRequest, DenseRole};
pub use moe::{MoeExecution, MoePlan, MoePlanRequest, MoeQuantization};
pub use output::{OutputHeadExecution, OutputHeadPlan, OutputHeadPlanRequest};

/// Inference phase used as part of an execution-plan key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ExecutionPhase {
    Decode,
    Prefill,
}

/// Numerical admission level for optimized kernels.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaNumericalPolicy {
    /// Admit only candidates that passed the operation's numerical gate.
    #[default]
    Validated,
    /// Permit separately marked throughput-first candidates.
    Throughput,
}

/// Stability level accepted by the execution planner.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaKernelAdmission {
    #[default]
    Stable,
    Experimental,
}

/// Explicit policy supplied by a CUDA library consumer.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct CudaPlanningPolicy {
    pub attention: CudaAttentionPolicy,
    pub numerical: CudaNumericalPolicy,
    pub admission: CudaKernelAdmission,
    pub dense_vectors: CudaDenseVectorPolicy,
    pub dense_vendor: CudaDenseVendorPolicy,
    pub dense_weights: CudaDenseWeightPolicy,
    pub moe_fusion: CudaMoeFusionPolicy,
    pub moe_batch: CudaMoeBatchPolicy,
    pub output_head: CudaOutputHeadPolicy,
}

/// Paged decode-attention selection policy.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaAttentionPolicy {
    #[default]
    Auto,
    Direct,
    SplitKv {
        partition_tokens: usize,
        threshold_tokens: usize,
    },
}

/// Experimental weight-only quantization selected for one dense role.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaDenseWeightPolicy {
    #[default]
    Bf16,
    BlockFp8Role(DenseRole),
    Fp8Int4Role(DenseRole),
}

/// Experimental bandwidth-oriented decode-vector candidates admitted by policy.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaDenseVectorPolicy {
    #[default]
    Disabled,
    Tuned,
    Role(DenseRole),
}

/// Experimental vendor-library candidates admitted for measured dense roles.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaDenseVendorPolicy {
    #[default]
    Disabled,
    Tuned,
    Role(DenseRole),
}

/// Experimental routed-MoE fusion candidates admitted by policy.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaMoeFusionPolicy {
    #[default]
    Disabled,
    Tuned,
}

/// Small-batch routed-MoE numerical format.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaMoeBatchPolicy {
    #[default]
    Auto,
    W4A4,
    /// Execute the experimental direct `SM12x` micro kernel for small decode
    /// batches.
    W4A4Direct,
    /// Retain Tensor Core gate/up and execute a direct activation/down tail.
    W4A4Hybrid,
    /// Group decode assignments by expert before native W4A4 execution.
    W4A4Bucketed,
    /// Force exact weight-only NVFP4 execution for every phase and batch size.
    W4A16,
}

/// Storage and execution policy for the decode output projection.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CudaOutputHeadPolicy {
    /// Select a validated implementation from hardware and projection geometry.
    #[default]
    Auto,
    Bf16,
    /// Quantize BF16 weights and activations to blockwise E4M3.
    Fp8Blockwise,
    /// Keep BF16 activations and execute over vectorized per-row E4M3 weights.
    Fp8Vectorized,
    /// Add a packed INT4 residual to vectorized per-row E4M3 weights.
    Fp8Residual,
    /// Keep BF16 activations and scale E4M3 weights in 128-element blocks.
    Fp8BlockVectorized,
    /// Refine block-scaled E4M3 top candidates with retained BF16 weights.
    Fp8BlockRefined,
}

/// Physical host/device memory relationship reported by CUDA.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaMemoryArchitecture {
    Unified,
    Discrete,
}

/// Model-independent hardware facts used by plan selection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CudaHardwareProfile {
    compute_capability: (u32, u32),
    multiprocessor_count: NonZeroU32,
    total_memory: NonZeroUsize,
    memory_architecture: CudaMemoryArchitecture,
}

impl CudaHardwareProfile {
    pub fn new(
        compute_capability: (u32, u32),
        multiprocessor_count: u32,
        total_memory: usize,
        memory_architecture: CudaMemoryArchitecture,
    ) -> Result<Self> {
        if compute_capability.0 == 0 {
            return Err(Error::InvalidExecutionPlan("CUDA compute capability is missing"));
        }
        Ok(Self {
            compute_capability,
            multiprocessor_count: NonZeroU32::new(multiprocessor_count)
                .ok_or(Error::InvalidExecutionPlan("CUDA device has no SMs"))?,
            total_memory: NonZeroUsize::new(total_memory)
                .ok_or(Error::InvalidExecutionPlan("CUDA device has no memory"))?,
            memory_architecture,
        })
    }

    pub(super) fn from_device(device: &DeviceInfo) -> Result<Self> {
        Self::new(
            (
                u32::try_from(device.compute_capability.0)?,
                u32::try_from(device.compute_capability.1)?,
            ),
            device.multiprocessor_count,
            device.total_memory,
            if device.integrated {
                CudaMemoryArchitecture::Unified
            } else {
                CudaMemoryArchitecture::Discrete
            },
        )
    }

    #[must_use]
    pub const fn compute_capability(self) -> (u32, u32) {
        self.compute_capability
    }

    #[must_use]
    pub const fn multiprocessor_count(self) -> NonZeroU32 {
        self.multiprocessor_count
    }

    #[must_use]
    pub const fn total_memory(self) -> NonZeroUsize {
        self.total_memory
    }

    #[must_use]
    pub const fn memory_architecture(self) -> CudaMemoryArchitecture {
        self.memory_architecture
    }
}

/// Origin of a selected execution strategy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PlanSource {
    ExplicitPolicy,
    Heuristic,
    MeasuredCache,
    MeasuredStartup,
    Fallback,
}

/// Pure, allocation-free selector for prepared CUDA execution strategies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaExecutionPlanner {
    hardware: CudaHardwareProfile,
    policy: CudaPlanningPolicy,
}

impl CudaExecutionPlanner {
    #[must_use]
    pub const fn new(hardware: CudaHardwareProfile, policy: CudaPlanningPolicy) -> Self {
        Self { hardware, policy }
    }

    #[must_use]
    pub const fn hardware(self) -> CudaHardwareProfile {
        self.hardware
    }

    #[must_use]
    pub const fn policy(self) -> CudaPlanningPolicy {
        self.policy
    }
}
pub use attention::{AttentionExecution, AttentionPlan, AttentionPlanRequest};
