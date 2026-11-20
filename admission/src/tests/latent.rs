use models::{
    execution::TaskExecutionPlan, layout::DecoderConfig, semantic::SemanticModelSpec,
    weights::TensorCatalog,
};
use serde_json::json;

use super::fixtures::AnyResult;
use crate::{cuda, metal};

/// A miniature DeepSeek-V2-Lite decoder: latent attention, one dense layer,
/// then routed experts with shared experts, routed by `scoring`.
fn latent_contract(scoring: &str) -> AnyResult<(TaskExecutionPlan, SemanticModelSpec)> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 2,
        "first_k_dense_replace": 1,
        "scoring_func": scoring,
        "topk_method": if scoring == "sigmoid" { "noaux_tc" } else { "greedy" },
        "norm_topk_prob": scoring == "sigmoid",
        "hidden_act": "silu"
    }))?;
    let semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    Ok((TaskExecutionPlan::Generation { decoder }, semantic))
}

#[test]
fn admits_latent_attention_on_cuda_and_refuses_it_on_metal() -> AnyResult<()> {
    let (task, semantic) = latent_contract("softmax")?;
    let plan = cuda::CudaDecoderPlan::lower(&semantic);
    assert!(plan.has_latent_mixer());
    assert_eq!(plan.layers()[1].mixer, cuda::CudaMixerLowering::Latent);
    assert_eq!(
        cuda::admit(&task, Some(&semantic))?,
        cuda::CudaArchitecture::Generation(cuda::CudaDecoderRuntime::LatentAttention)
    );
    let refusal = metal::admit(&task, Some(&semantic)).map(|_| ()).err();
    let message = refusal.map(|error| error.to_string()).unwrap_or_default();
    assert!(message.contains("multi-head latent attention"), "unexpected refusal: {message}");
    Ok(())
}

#[test]
fn refuses_a_sigmoid_router_by_name() -> AnyResult<()> {
    let (task, semantic) = latent_contract("sigmoid")?;
    let message = cuda::admit(&task, Some(&semantic))
        .map(|_| ())
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(message.contains("SigmoidBiasedTopK"), "unexpected refusal: {message}");
    Ok(())
}
