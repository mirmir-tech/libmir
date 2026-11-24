use models::{
    execution::TaskExecutionPlan, layout::DecoderConfig, semantic::SemanticModelSpec,
    weights::TensorCatalog,
};
use serde_json::json;

use super::fixtures::AnyResult;
use crate::{
    ArchitectureError, Unlowerable,
    capability::FeedForwardShape,
    cuda,
    lowering::{FeedForwardLowering, MixerLowering},
    metal,
};

pub(super) fn hybrid_contract() -> AnyResult<(TaskExecutionPlan, SemanticModelSpec)> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 4,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_latent_size": 16,
        "moe_intermediate_size": 32,
        "norm_topk_prob": true,
        "mlp_hidden_act": "relu2"
    }))?;
    let semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    Ok((TaskExecutionPlan::Generation { decoder }, semantic))
}

#[test]
fn lowers_state_space_hybrid_layers_on_cuda() -> AnyResult<()> {
    let (_, semantic) = hybrid_contract()?;
    let plan = cuda::CudaDecoderPlan::lower(&semantic, None)?;
    assert_eq!(plan.layers()[0].mixer, MixerLowering::StateSpace);
    assert_eq!(plan.layers()[0].feed_forward.shape(), FeedForwardShape::LatentRouted);
    assert_eq!(plan.layers()[1].feed_forward, FeedForwardLowering::None);
    Ok(())
}

#[test]
fn cuda_rejects_routed_experts_without_a_latent_space() -> AnyResult<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 4,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_latent_size": null,
        "moe_intermediate_size": 32,
        "norm_topk_prob": true,
        "mlp_hidden_act": "relu2"
    }))?;
    let semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    let task = TaskExecutionPlan::Generation { decoder };
    assert_eq!(
        cuda::admit(&task, Some(&semantic)).err(),
        Some(ArchitectureError::Unlowerable {
            layer: Some(0),
            feature: Unlowerable::LatentExpertsWithoutLatentSize,
        })
    );
    assert_eq!(
        metal::admit(&task, Some(&semantic))?.runtime(),
        Some(metal::MetalDecoderRuntime::StateSpaceHybrid)
    );
    Ok(())
}
