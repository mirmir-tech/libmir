//! CUDA architecture admission: lower, require, select.

mod lowering;
mod runtime;

pub use lowering::{CudaDecoderPlan, CudaLayer, RepeatedLayer, YarnRotary};
use models::{
    execution::TaskExecutionPlan,
    layout::{EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind},
    semantic::SemanticModelSpec,
};
pub use runtime::CudaDecoderRuntime;

use crate::{ArchitectureError, Result, TaskRefusal};

/// A generation plan and the runtime admitted to execute it, which the
/// loaders build from.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaAdmitted {
    pub runtime: CudaDecoderRuntime,
    pub plan: CudaDecoderPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CudaArchitecture {
    Generation(CudaAdmitted),
    Embedding,
    SequenceScoring,
}

impl CudaArchitecture {
    /// The admitted generation runtime.
    #[must_use]
    pub const fn runtime(&self) -> Option<CudaDecoderRuntime> {
        match self {
            Self::Generation(admitted) => Some(admitted.runtime),
            Self::Embedding | Self::SequenceScoring => None,
        }
    }
}

pub fn admit(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<CudaArchitecture> {
    match task {
        TaskExecutionPlan::Generation { decoder }
        | TaskExecutionPlan::CausalScoring { decoder, .. } => {
            let semantic = semantic.ok_or(ArchitectureError::MissingSemanticContract)?;
            let plan = CudaDecoderPlan::lower(semantic, decoder.attention_scale)?;
            let runtime = CudaDecoderRuntime::select(&plan)?;
            Ok(CudaArchitecture::Generation(CudaAdmitted { runtime, plan }))
        },
        TaskExecutionPlan::Embedding { .. } => Ok(CudaArchitecture::Embedding),
        TaskExecutionPlan::SequenceScoring { encoder, .. } => sequence_scoring(encoder),
    }
}

fn sequence_scoring(config: &EncoderConfig) -> Result<CudaArchitecture> {
    let fixed_ntk =
        matches!(config.rope_scaling, Some(EncoderRopeScaling::Ntk { mixed_b: None, .. }));
    if config.packed_qkv
        && config.norm == NormKind::LayerNorm
        && config.hidden_activation == "gelu"
        && config.position_embedding == EncoderPositionEmbedding::Rope
        && config.type_vocab_size > 0
        && config.num_labels == 1
        && fixed_ntk
    {
        Ok(CudaArchitecture::SequenceScoring)
    } else {
        Err(ArchitectureError::Task(TaskRefusal::SequenceScoringEncoder))
    }
}
