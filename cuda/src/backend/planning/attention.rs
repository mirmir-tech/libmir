use super::{CudaAttentionPolicy, CudaExecutionPlanner, PlanSource};
use crate::{Error, Result};

/// Complete shape key used to select paged decode attention.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttentionPlanRequest {
    pub max_context_tokens: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
}

/// Prepared attention strategy used by a stable CUDA graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttentionExecution {
    Direct,
    SplitKv {
        partition_tokens: usize,
        threshold_tokens: usize,
    },
}

/// Selected attention strategy and its provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttentionPlan {
    execution: AttentionExecution,
    source: PlanSource,
}

impl AttentionPlan {
    #[must_use]
    pub const fn execution(self) -> AttentionExecution {
        self.execution
    }

    #[must_use]
    pub const fn source(self) -> PlanSource {
        self.source
    }
}

impl CudaExecutionPlanner {
    pub fn plan_attention(self, request: AttentionPlanRequest) -> Result<AttentionPlan> {
        validate(request)?;
        let policy = self.policy().attention;
        let (execution, source) = match policy {
            CudaAttentionPolicy::Auto
                if self.hardware().compute_capability().0 == 12
                    && request.max_context_tokens >= 512
                    && request.head_dim >= 256
                    && request.value_head_dim >= 256 =>
            {
                (
                    AttentionExecution::SplitKv {
                        partition_tokens: 64,
                        threshold_tokens: 128,
                    },
                    PlanSource::Tuned,
                )
            },
            CudaAttentionPolicy::Auto
                if self.hardware().compute_capability().0 == 12
                    && request.max_context_tokens >= 512
                    && request.head_dim <= 128
                    && request.value_head_dim <= 128 =>
            {
                (
                    AttentionExecution::SplitKv {
                        partition_tokens: 64,
                        threshold_tokens: 65,
                    },
                    PlanSource::Tuned,
                )
            },
            CudaAttentionPolicy::Auto
                if self.hardware().compute_capability().0 == 12
                    && request.max_context_tokens >= 512 =>
            {
                (
                    AttentionExecution::SplitKv {
                        partition_tokens: 256,
                        threshold_tokens: 512,
                    },
                    PlanSource::Tuned,
                )
            },
            CudaAttentionPolicy::Auto => (AttentionExecution::Direct, PlanSource::Fallback),
            CudaAttentionPolicy::Direct => (AttentionExecution::Direct, PlanSource::ExplicitPolicy),
            CudaAttentionPolicy::SplitKv { partition_tokens, threshold_tokens } => {
                validate_split(partition_tokens, threshold_tokens)?;
                (
                    AttentionExecution::SplitKv { partition_tokens, threshold_tokens },
                    PlanSource::ExplicitPolicy,
                )
            },
        };
        tracing::debug!(
            target: "libmir::cuda::planning",
            compute_major = self.hardware().compute_capability().0,
            multiprocessors = self.hardware().multiprocessor_count().get(),
            policy = ?policy,
            max_context_tokens = request.max_context_tokens,
            query_heads = request.query_heads,
            kv_heads = request.kv_heads,
            head_dim = request.head_dim,
            value_head_dim = request.value_head_dim,
            execution = ?execution,
            source = ?source,
            "selected CUDA attention plan"
        );
        Ok(AttentionPlan { execution, source })
    }
}

fn validate(request: AttentionPlanRequest) -> Result<()> {
    if request.max_context_tokens == 0
        || request.max_context_tokens >= usize::try_from(u32::MAX)?
        || request.query_heads == 0
        || request.kv_heads == 0
        || !request.query_heads.is_multiple_of(request.kv_heads)
        || request.head_dim == 0
        || request.value_head_dim == 0
    {
        Err(Error::InvalidExecutionPlan("attention plan has invalid geometry"))
    } else {
        Ok(())
    }
}

fn validate_split(partition_tokens: usize, threshold_tokens: usize) -> Result<()> {
    if partition_tokens == 0 || threshold_tokens <= partition_tokens {
        Err(Error::InvalidExecutionPlan("split-KV attention policy has invalid limits"))
    } else {
        Ok(())
    }
}
