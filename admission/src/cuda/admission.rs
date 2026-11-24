use models::{
    execution::TaskExecutionPlan,
    layout::{EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind},
    semantic::{
        ActivationSpec, FeedForwardSpec, KeyValueRelation, QkNormalization, RouterNormalization,
        SemanticModelSpec,
    },
};

use super::{
    CudaArchitecture, CudaDecoderPlan, CudaDecoderRuntime, CudaFeedForwardLowering,
    CudaMixerLowering, require_graph_attention,
};
use crate::{ArchitectureError, Result};

pub fn admit(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<CudaArchitecture> {
    match task {
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. } => {
            generation(semantic)
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
        Err(ArchitectureError::invalid(
            "CUDA sequence scoring requires packed QKV, token types, LayerNorm, GELU, one label, and fixed NTK RoPE",
        ))
    }
}

fn generation(semantic: Option<&SemanticModelSpec>) -> Result<CudaArchitecture> {
    let semantic = semantic.ok_or_else(|| {
        ArchitectureError::invalid("CUDA generation semantic contract is missing")
    })?;
    crate::activation::require_resolved(semantic)?;
    if crate::uses_layer_sharing(semantic) {
        return Err(ArchitectureError::invalid(
            "CUDA does not run per-layer inputs or layers sharing keys and values",
        ));
    }
    let plan = CudaDecoderPlan::lower(semantic);
    let runtime = if plan.has_latent_mixer() {
        super::latent::admit(&plan, semantic)?
    } else if plan.has_state_space_mixer() {
        state_space_hybrid(&plan, semantic)?
    } else if plan.all_shared_routed() && plan.has_linear_mixer() && plan.has_softmax_mixer() {
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
    } else if plan.all_dense_or_routed() {
        require_graph_attention(&plan)?;
        CudaDecoderRuntime::Dense
    } else {
        return Err(ArchitectureError::invalid(format!(
            "CUDA has no runtime for the {}-layer decoder composition",
            plan.layers().len()
        )));
    };
    super::dense::require_graph_decoder(semantic, runtime)?;
    Ok(CudaArchitecture::Generation(runtime))
}

/// Mamba-2 and attention mixers, without Q/K normalization, sinks, or a
/// linear-attention layer; latent routed experts, an ungated `relu²` MLP or
/// no feed-forward.
fn state_space_hybrid(
    plan: &CudaDecoderPlan,
    semantic: &SemanticModelSpec,
) -> Result<CudaDecoderRuntime> {
    if semantic.decoder.layers.iter().any(|layer| {
        matches!(layer.feed_forward, FeedForwardSpec::LatentRouted { latent_size: None, .. })
    }) {
        return Err(ArchitectureError::invalid(
            "CUDA state-space hybrid runtime requires routed experts in a projected latent space; this checkpoint declares no moe_latent_size",
        ));
    }
    // The latent experts' router is the biased sigmoid one, and their MLPs
    // the ungated relu² ones.
    if semantic.decoder.layers.iter().any(|layer| {
        matches!(&layer.feed_forward, FeedForwardSpec::LatentRouted { routed, .. }
            if routed.router_normalization != RouterNormalization::SigmoidBiasedTopK
                || routed.activation != ActivationSpec::ReluSquared)
    }) {
        return Err(ArchitectureError::invalid(
            "CUDA latent experts route by biased sigmoid into relu² experts",
        ));
    }
    if semantic.decoder.layers.iter().any(|layer| {
        matches!(&layer.feed_forward, FeedForwardSpec::Dense { activation, .. }
            if *activation != ActivationSpec::ReluSquared)
    }) {
        return Err(ArchitectureError::invalid(
            "CUDA state-space hybrid runtime runs only ungated relu² MLP blocks",
        ));
    }
    let supported = plan.layers().iter().all(|layer| {
        let mixer = match layer.mixer {
            CudaMixerLowering::StateSpace => true,
            CudaMixerLowering::Softmax { sinks, normalization, key_value_relation } => {
                !sinks
                    && normalization == QkNormalization::None
                    && key_value_relation == KeyValueRelation::Separate
            },
            CudaMixerLowering::Linear | CudaMixerLowering::Latent => false,
        };
        mixer
            && matches!(
                layer.feed_forward,
                CudaFeedForwardLowering::LatentRouted { .. }
                    | CudaFeedForwardLowering::Dense
                    | CudaFeedForwardLowering::None
            )
    });
    if supported && plan.has_softmax_mixer() {
        Ok(CudaDecoderRuntime::StateSpaceHybrid)
    } else {
        Err(ArchitectureError::invalid(
            "CUDA state-space hybrid runtime requires Mamba-2 and plain attention mixers with latent routed, relu² MLP or no feed-forward",
        ))
    }
}
