use models::{
    execution::TaskExecutionPlan, layout::DecoderConfig, semantic::SemanticModelSpec,
    weights::TensorCatalog,
};
use serde_json::json;

use super::fixtures::AnyResult;
use crate::{cuda, metal};

fn hybrid_contract() -> AnyResult<(TaskExecutionPlan, SemanticModelSpec)> {
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
    let plan = cuda::CudaDecoderPlan::lower(&semantic);

    assert!(plan.has_state_space_mixer());
    assert_eq!(plan.layers()[0].mixer, cuda::CudaMixerLowering::StateSpace);
    assert_eq!(
        plan.layers()[0].feed_forward,
        cuda::CudaFeedForwardLowering::LatentRouted { shared: false }
    );
    assert_eq!(plan.layers()[1].feed_forward, cuda::CudaFeedForwardLowering::None);
    Ok(())
}

#[test]
fn admits_state_space_hybrids_on_cuda_only() -> AnyResult<()> {
    let (task, semantic) = hybrid_contract()?;

    assert_eq!(
        cuda::admit(&task, Some(&semantic))?,
        cuda::CudaArchitecture::Generation(cuda::CudaDecoderRuntime::StateSpaceHybrid)
    );
    let metal = metal::admit(&task, Some(&semantic)).err().map(|error| error.to_string());
    assert!(metal.is_some_and(|error| error.contains("state-space mixer")));
    Ok(())
}
