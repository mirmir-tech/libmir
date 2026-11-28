use models::layout::DecoderConfig;
use serde_json::json;

use super::*;

mod chunked;
mod vision;

#[test]
fn reserves_at_least_half_a_gibibyte_for_runtime_workspace() {
    assert_eq!(workspace(128 * MIB), 512 * MIB);
    assert_eq!(workspace(10 * 1024 * MIB), 1024 * MIB);
}

#[test]
fn accounts_for_persistent_marlin_weights_scales_and_globals() {
    assert_eq!(nvfp4_marlin_bytes(&[4, 32, 64]), 4_624);
    assert_eq!(nvfp4_marlin_bytes(&[32, 64]), 1_156);
}

#[test]
fn bounds_clamped_sliding_layers_per_concurrent_session() -> crate::Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "gpt_oss",
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "vocab_size": 256,
        "max_position_embeddings": 131_072,
        "num_local_experts": 4,
        "experts_per_token": 2,
        "sliding_window": 128,
        "swiglu_limit": 7.0,
        "attention_bias": true,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096
        },
        "layer_types": ["sliding_attention", "full_attention"]
    }))?;
    let config = RuntimeConfig::default();
    let bounded = ring_cache(128, architecture::CUDA_SLIDING_SLOTS_PER_SESSION, &config);
    let metal = ring_cache(128, 1, &config);
    let (growing, total) = decoder_budget_with_tensors(
        &decoder,
        &[
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.sinks",
            "model.layers.1.self_attn.sinks",
        ],
        &config,
        BackendTarget::Cuda,
    )?;

    assert_eq!(bounded.block_count, 576);
    assert_eq!(metal.block_count, 144);
    assert_eq!(growing, 256);
    assert!(total < growing * 4_096 * 16 * 2);
    Ok(())
}

#[test]
fn prices_a_latent_layer_as_its_latent_and_rotary_key() -> crate::Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 2048,
        "intermediate_size": 10_944,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 102_400,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
    }))?;
    let (growing, _) = decoder_budget(&decoder, &RuntimeConfig::default(), BackendTarget::Cuda)?;

    // BF16: 27 layers of one 512-wide latent plus one 64-wide rotary key, not
    // 16 heads of 192-wide keys and 128-wide values.
    assert_eq!(growing, 27 * (512 + 64) * 2);
    Ok(())
}

#[test]
fn prices_mamba_state_and_convolution_history_per_sequence() -> crate::Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 4096,
        "intermediate_size": 2688,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 131_072,
        "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "chunk_size": 128,
        "mamba_ssm_cache_dtype": "float32",
        "n_routed_experts": 512,
        "num_experts_per_tok": 22,
        "moe_latent_size": 1024,
        "moe_intermediate_size": 2688,
        "norm_topk_prob": true,
        "mlp_hidden_act": "relu2"
    }))?;
    let state = 128 * 64 * 128 * 4;
    let history = 10_240 * 3 * 2;

    assert_eq!(session::state_bytes(&decoder, 0), 2 * (state + history));
    // Three rollback snapshots for three drafts.
    assert_eq!(session::state_bytes(&decoder, 3), 8 * (state + history));
    let config = RuntimeConfig::default();
    let (per_token, _) = decoder_budget(&decoder, &config, BackendTarget::Cuda)?;
    let attention_only = KvStorageSpec::new(config.kv_cache, 2, 128).memory_budget();
    assert_eq!(
        per_token,
        u64::try_from(attention_only.data_bytes_per_token + attention_only.scale_bytes_per_token)
            .unwrap_or(u64::MAX)
    );
    Ok(())
}

#[test]
fn prediction_cache_and_rollback_residency_follow_backend_drafts() -> crate::Result<()> {
    use models::{
        semantic::SemanticModelSpec,
        weights::{TensorCatalog, WeightBindingPlan},
    };
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size":128,"intermediate_size":256,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,
        "vocab_size":256,"max_position_embeddings":1024
    }))?;
    let mut semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: vec![] })?;
    semantic.decoder.prediction_layers = semantic.decoder.layers.clone();
    let execution = DecoderExecutionContract {
        semantic,
        bindings: WeightBindingPlan { tensors: vec![] },
    };
    let mut config = RuntimeConfig::default();
    config.scheduler.speculation =
        Some(crate::Speculation { draft_tokens: crate::DraftTokens::MAX });
    let metal = BackendTarget::Metal;
    let metal_layout = CacheLayout::for_metal(&execution.semantic.decoder);
    let cuda_layout = CacheLayout::for_cuda(&execution.semantic.decoder, CudaDecoderRuntime::Dense);
    assert!(prediction::kv_budget(&metal_layout.prediction_layers, &config, metal).0 > 0);
    assert_eq!(prediction::rollback_snapshots(Some(&execution), &config, metal), 8);
    let (per_token, total) =
        prediction::kv_budget(&cuda_layout.prediction_layers, &config, BackendTarget::Cuda);
    assert!(per_token > 0 && total > per_token);
    assert_eq!(
        prediction::rollback_snapshots(Some(&execution), &config, BackendTarget::Cuda),
        8
    );
    config.scheduler.speculation = None;
    assert_eq!(
        prediction::kv_budget(&cuda_layout.prediction_layers, &config, BackendTarget::Cuda),
        (0, 0)
    );
    Ok(())
}

/// The decoder's K/V budget through its semantic specification.
fn decoder_budget(
    decoder: &DecoderConfig,
    config: &RuntimeConfig,
    target: BackendTarget,
) -> crate::Result<(u64, u64)> {
    decoder_budget_with_tensors(decoder, &["model.layers.0.input_layernorm.weight"], config, target)
}

fn decoder_budget_with_tensors(
    decoder: &DecoderConfig,
    tensor_names: &[&str],
    config: &RuntimeConfig,
    target: BackendTarget,
) -> crate::Result<(u64, u64)> {
    let catalog = models::weights::TensorCatalog {
        tensors: tensor_names
            .iter()
            .map(|name| models::weights::TensorInfo {
                name: (*name).into(),
                file: std::path::PathBuf::new(),
                container: models::layout::WeightContainer::SafeTensors,
                dtype: models::weights::DenseDType::Bf16.into(),
                shape: vec![decoder.hidden_size],
                data_start: 0,
                data_offsets: [0, 0],
            })
            .collect(),
    };
    let spec = SemanticModelSpec::discover(decoder, &catalog)?;
    let task = TaskExecutionPlan::Generation { decoder: decoder.clone() };
    let layout = cache_layout(&task, &spec, target);
    Ok(kv_budget(&layout.layers, config))
}
