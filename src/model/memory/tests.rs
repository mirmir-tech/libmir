use serde_json::json;

use super::*;

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
        "layer_types": ["sliding_attention", "full_attention"]
    }))?;
    let config = RuntimeConfig::default();
    let bounded = sliding_cache(&decoder, &config, &BackendTarget::Cuda);
    let metal = sliding_cache(&decoder, &config, &BackendTarget::Metal);
    let (growing, total) = kv_budget(&decoder, &config, &BackendTarget::Cuda);

    assert_eq!(bounded.block_count, 576);
    assert_eq!(metal.block_count, 144);
    assert_eq!(growing, 256);
    assert!(total < growing * 4_096 * 16 * 2);
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

    assert_eq!(session::state_bytes(&decoder), 2 * (state + history));
    let config = RuntimeConfig::default();
    let (per_token, _) = kv_budget(&decoder, &config, &BackendTarget::Cuda);
    let attention_only = KvStorageSpec::new(config.kv_cache, 2, 128).memory_budget();
    assert_eq!(
        per_token,
        u64::try_from(attention_only.data_bytes_per_token + attention_only.scale_bytes_per_token)
            .unwrap_or(u64::MAX)
    );
    Ok(())
}
