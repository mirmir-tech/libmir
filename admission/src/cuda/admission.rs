use models::{
    execution::TaskExecutionPlan,
    layout::{EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind},
    semantic::SemanticModelSpec,
};

use super::{
    CudaArchitecture, CudaDecoderPlan, CudaDecoderRuntime, CudaFeedForwardLowering,
    graph_normalization,
};
use crate::{ArchitectureError, Result};

pub fn admit(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<CudaArchitecture> {
    match task {
        TaskExecutionPlan::Generation { .. } => generation(semantic),
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
        Err(ArchitectureError::invalid(
            "CUDA sequence scoring requires packed QKV, token types, LayerNorm, GELU, one label, and fixed NTK RoPE",
        ))
    }
}

fn generation(semantic: Option<&SemanticModelSpec>) -> Result<CudaArchitecture> {
    let semantic = semantic.ok_or_else(|| {
        ArchitectureError::invalid("CUDA generation semantic contract is missing")
    })?;
    let plan = CudaDecoderPlan::lower(semantic);
    let runtime = if plan.all_shared_routed() && plan.has_linear_mixer() && plan.has_softmax_mixer()
    {
        CudaDecoderRuntime::SharedRouted
    } else if plan.has_linear_mixer()
        && plan.has_softmax_mixer()
        && plan
            .layers()
            .iter()
            .all(|layer| layer.feed_forward == CudaFeedForwardLowering::Dense)
    {
        CudaDecoderRuntime::DenseMixed
    } else if plan.all_unshared_clamped_routed() {
        CudaDecoderRuntime::ClampedRouted
    } else if plan.all_dense_and_routed() {
        CudaDecoderRuntime::DenseAndRouted
    } else if plan.all_dense() {
        let _normalization = graph_normalization(&plan)?;
        CudaDecoderRuntime::Dense
    } else {
        return Err(ArchitectureError::invalid(format!(
            "CUDA has no runtime for the {}-layer decoder composition",
            plan.layers().len()
        )));
    };
    Ok(CudaArchitecture::Generation(runtime))
}
