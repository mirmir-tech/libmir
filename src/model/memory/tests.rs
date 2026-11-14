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

    assert_eq!(session::state_bytes(&decoder, 0), 2 * (state + history));
    // Three rollback snapshots for three drafts.
    assert_eq!(session::state_bytes(&decoder, 3), 8 * (state + history));
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

#[test]
fn prediction_cache_and_rollback_residency_are_cuda_only() -> crate::Result<()> {
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
    for target in [BackendTarget::Metal, BackendTarget::CpuReference] {
        assert_eq!(prediction::kv_budget(Some(&execution), &config, &target), (0, 0));
        assert_eq!(prediction::rollback_snapshots(Some(&execution), &config, &target), 0);
    }
    let (per_token, total) = prediction::kv_budget(Some(&execution), &config, &BackendTarget::Cuda);
    assert!(per_token > 0 && total > per_token);
    assert_eq!(
        prediction::rollback_snapshots(Some(&execution), &config, &BackendTarget::Cuda),
        8
    );
    config.scheduler.speculation = None;
    assert_eq!(prediction::kv_budget(Some(&execution), &config, &BackendTarget::Cuda), (0, 0));
    Ok(())
}

#[test]
fn a_vision_load_still_leaves_prediction_modules_on_disk() -> crate::Result<()> {
    use std::path::PathBuf;

    use models::{
        layout::VisionConfig,
        weights::{
            PredictionTensorRole, TensorBinding, TensorCatalog, TensorInfo, WeightBindingPlan,
        },
    };
    let binding = |source: &str, role| TensorBinding {
        role,
        source: source.into(),
        shape: vec![8],
        logical_shape: None,
        transforms: Vec::new(),
        storage: TensorStorage::Dense { dtype: "BF16".into(), bias: None },
    };
    let prediction = LogicalTensorRole::Prediction {
        depth: 0,
        tensor: PredictionTensorRole::EmbeddingNorm,
    };
    let auxiliary = LogicalTensorRole::Auxiliary { path: "vision".into() };
    let bindings = WeightBindingPlan {
        tensors: vec![binding("mtp.enorm.weight", prediction), binding("vision.weight", auxiliary)],
    };
    let tensor = |name: &str, bytes: u64| TensorInfo {
        name: name.into(),
        file: PathBuf::from("model.safetensors"),
        dtype: "BF16".into(),
        shape: vec![8],
        data_start: 0,
        data_offsets: [0, bytes],
    };
    let catalog = TensorCatalog {
        tensors: vec![tensor("mtp.enorm.weight", 16), tensor("vision.weight", 64)],
    };
    let vision = VisionConfig::from_value(&pixel_shuffle_config())?
        .ok_or_else(|| models::ModelsError::InvalidConfig("vision not discovered".into()))?;
    let counts = |unloaded: Option<UnloadedTensors>| {
        unloaded.map(|unloaded| (unloaded.tensors, unloaded.bytes, unloaded.prediction_bytes))
    };
    assert_eq!(counts(unloaded(&catalog, Some(&bindings), None)), Some((2, 80, 16)));
    // The tower loads alongside the decoder; the prediction module does not.
    assert_eq!(counts(unloaded(&catalog, Some(&bindings), Some(&vision))), Some((1, 16, 16)));
    Ok(())
}

fn pixel_shuffle_config() -> serde_json::Value {
    json!({
        "downsample_ratio": 0.5,
        "ps_version": "v2",
        "vit_hidden_size": 64,
        "projector_hidden_size": 256,
        "img_context_token_id": 18,
        "image_tag_type": "internvl",
        "llm_config": {"hidden_size": 128, "num_nextn_predict_layers": 1},
        "vision_config": {
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 16,
            "mlp_ratio": 4.0,
            "num_cls_tokens": 1,
            "num_registers": 0,
            "max_img_size": 256,
            "layer_norm_eps": 1e-06,
            "hidden_act": "gelu"
        }
    })
}
