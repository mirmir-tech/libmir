//! Dense checkpoints Metal admits by the binding plan of the runtime it
//! selects: routed experts without sinks run on the dense runtime, and
//! latent attention with leading dense layers on the latent one. Both were
//! refused by the plans of other runtimes.

use super::*;

const HIDDEN: usize = 32;

fn admitted_dense(config: &serde_json::Value, catalog: &TensorCatalog) -> Result<()> {
    let contract = RemoteModelContract::inspect_generation(config, catalog)?;
    let metal = contract.admission(BackendTarget::Metal);
    assert_eq!(metal.status, AdmissionStatus::Supported, "{metal:?}");
    assert!(metal.checks.iter().any(|check| {
        check.kind() == AdmissionCheckKind::Encoding(EncodingKind::Dense)
            && check.status() == AdmissionStatus::Supported
    }));
    Ok(())
}

fn add(tensors: &mut Vec<TensorInfo>, name: &str, shape: &[usize]) {
    tensors.push(tensor_with_dtype(name, shape.to_vec(), "BF16"));
}

fn experts(tensors: &mut Vec<TensorInfo>, mlp: &str, count: usize, width: usize) {
    for expert in 0..count {
        for (projection, shape) in
            [("gate", [width, HIDDEN]), ("up", [width, HIDDEN]), ("down", [HIDDEN, width])]
        {
            add(tensors, &format!("{mlp}.experts.{expert}.{projection}_proj.weight"), &shape);
        }
    }
}

/// Qwen3-MoE: routed experts without a shared expert or attention sinks.
#[test]
fn admits_dense_routed_experts_without_sinks_on_metal() -> Result<()> {
    let config = json!({
        "model_type": "qwen3_moe", "hidden_size": HIDDEN, "intermediate_size": 16,
        "moe_intermediate_size": 8, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64, "num_experts": 4,
        "num_experts_per_tok": 2, "norm_topk_prob": true, "decoder_sparse_step": 1,
        "mlp_only_layers": [], "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false
    });
    let mut tensors = Vec::new();
    add(&mut tensors, "model.embed_tokens.weight", &[64, HIDDEN]);
    add(&mut tensors, "model.norm.weight", &[HIDDEN]);
    add(&mut tensors, "lm_head.weight", &[64, HIDDEN]);
    let layer = "model.layers.0";
    add(&mut tensors, &format!("{layer}.input_layernorm.weight"), &[HIDDEN]);
    add(&mut tensors, &format!("{layer}.post_attention_layernorm.weight"), &[HIDDEN]);
    for (projection, shape) in
        [("q", [32, HIDDEN]), ("k", [16, HIDDEN]), ("v", [16, HIDDEN]), ("o", [HIDDEN, 32])]
    {
        add(&mut tensors, &format!("{layer}.self_attn.{projection}_proj.weight"), &shape);
    }
    add(&mut tensors, &format!("{layer}.self_attn.q_norm.weight"), &[8]);
    add(&mut tensors, &format!("{layer}.self_attn.k_norm.weight"), &[8]);
    add(&mut tensors, &format!("{layer}.mlp.gate.weight"), &[4, HIDDEN]);
    experts(&mut tensors, &format!("{layer}.mlp"), 4, 8);
    admitted_dense(&config, &TensorCatalog::new(tensors))
}

/// DeepSeek-V2: latent attention, one leading dense layer, then routed plus
/// shared experts.
#[test]
fn admits_dense_latent_attention_experts_on_metal() -> Result<()> {
    let config = json!({
        "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
        "hidden_size": HIDDEN, "intermediate_size": 16, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "vocab_size": 64, "q_lora_rank": null, "kv_lora_rank": 16, "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4, "v_head_dim": 8, "n_routed_experts": 4,
        "num_experts_per_tok": 2, "n_shared_experts": 2, "first_k_dense_replace": 1,
        "moe_layer_freq": 1, "n_group": 1, "topk_group": 1, "scoring_func": "softmax",
        "topk_method": "greedy", "norm_topk_prob": false, "routed_scaling_factor": 1.0,
        "hidden_act": "silu", "rms_norm_eps": 1e-6
    });
    let mut tensors = Vec::new();
    add(&mut tensors, "model.embed_tokens.weight", &[64, HIDDEN]);
    add(&mut tensors, "model.norm.weight", &[HIDDEN]);
    add(&mut tensors, "lm_head.weight", &[64, HIDDEN]);
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        add(&mut tensors, &format!("{prefix}.input_layernorm.weight"), &[HIDDEN]);
        add(&mut tensors, &format!("{prefix}.post_attention_layernorm.weight"), &[HIDDEN]);
        add(&mut tensors, &format!("{attention}.q_proj.weight"), &[4 * 12, HIDDEN]);
        add(&mut tensors, &format!("{attention}.kv_a_proj_with_mqa.weight"), &[20, HIDDEN]);
        add(&mut tensors, &format!("{attention}.kv_a_layernorm.weight"), &[16]);
        add(&mut tensors, &format!("{attention}.kv_b_proj.weight"), &[4 * 16, 16]);
        add(&mut tensors, &format!("{attention}.o_proj.weight"), &[HIDDEN, 4 * 8]);
        let mlp = format!("{prefix}.mlp");
        if layer == 0 {
            for (projection, shape) in
                [("gate", [16, HIDDEN]), ("up", [16, HIDDEN]), ("down", [HIDDEN, 16])]
            {
                add(&mut tensors, &format!("{mlp}.{projection}_proj.weight"), &shape);
            }
            continue;
        }
        add(&mut tensors, &format!("{mlp}.gate.weight"), &[4, HIDDEN]);
        for (projection, shape) in
            [("gate", [16, HIDDEN]), ("up", [16, HIDDEN]), ("down", [HIDDEN, 16])]
        {
            add(&mut tensors, &format!("{mlp}.shared_experts.{projection}_proj.weight"), &shape);
        }
        experts(&mut tensors, &mlp, 4, 8);
    }
    admitted_dense(&config, &TensorCatalog::new(tensors))
}

/// A dense decoder Metal does not lower is refused by its architecture
/// check alone: without routed experts its dense storage is not what
/// fails, so that check does not blame a routed composition.
#[test]
fn a_dense_model_metal_does_not_lower_is_refused_by_architecture_only() -> Result<()> {
    let config = json!({
        "model_type": "llama", "hidden_size": HIDDEN, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "vocab_size": 64, "hidden_act": "unknown_gelu", "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false
    });
    let mut tensors = Vec::new();
    add(&mut tensors, "model.embed_tokens.weight", &[64, HIDDEN]);
    add(&mut tensors, "model.norm.weight", &[HIDDEN]);
    add(&mut tensors, "lm_head.weight", &[64, HIDDEN]);
    let layer = "model.layers.0";
    add(&mut tensors, &format!("{layer}.input_layernorm.weight"), &[HIDDEN]);
    add(&mut tensors, &format!("{layer}.post_attention_layernorm.weight"), &[HIDDEN]);
    for (projection, shape) in
        [("q", [32, HIDDEN]), ("k", [16, HIDDEN]), ("v", [16, HIDDEN]), ("o", [HIDDEN, 32])]
    {
        add(&mut tensors, &format!("{layer}.self_attn.{projection}_proj.weight"), &shape);
    }
    for (projection, shape) in
        [("gate", [16, HIDDEN]), ("up", [16, HIDDEN]), ("down", [HIDDEN, 16])]
    {
        add(&mut tensors, &format!("{layer}.mlp.{projection}_proj.weight"), &shape);
    }
    let contract = RemoteModelContract::inspect_generation(&config, &TensorCatalog::new(tensors))?;
    let metal = contract.admission(BackendTarget::Metal);
    assert_eq!(metal.status, AdmissionStatus::Unsupported, "{metal:?}");
    let status = |kind| {
        metal
            .checks
            .iter()
            .find(|check| check.kind() == kind)
            .map(crate::AdmissionCheck::status)
    };
    assert_eq!(status(AdmissionCheckKind::Architecture), Some(AdmissionStatus::Unsupported));
    assert_eq!(
        status(AdmissionCheckKind::Encoding(EncodingKind::Dense)),
        Some(AdmissionStatus::Supported),
        "{metal:?}"
    );
    Ok(())
}

/// Nemotron-H: mixer-only Mamba-2 and attention layers have
/// `FeedForwardSpec::None`, which must pass dense admission alongside routed
/// layers.
#[test]
fn admits_dense_state_space_hybrid_with_mixer_only_layers_on_metal() -> Result<()> {
    let config = json!({
        "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h",
        "hidden_size": HIDDEN, "intermediate_size": 8, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64, "layer_norm_epsilon": 1e-5,
        "layers_block_type": ["mamba", "attention", "moe"], "mamba_num_heads": 4,
        "mamba_head_dim": 16, "ssm_state_size": 32, "n_groups": 2, "conv_kernel": 4,
        "chunk_size": 8, "expand": 2, "mamba_ssm_cache_dtype": "float32", "use_conv_bias": true,
        "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_latent_size": null,
        "moe_intermediate_size": 8, "moe_shared_expert_intermediate_size": 16,
        "n_shared_experts": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": true,
        "mlp_hidden_act": "relu2", "tie_word_embeddings": false
    });
    let mut tensors = Vec::new();
    add(&mut tensors, "backbone.embeddings.weight", &[64, HIDDEN]);
    add(&mut tensors, "backbone.norm_f.weight", &[HIDDEN]);
    add(&mut tensors, "lm_head.weight", &[64, HIDDEN]);
    add(&mut tensors, "backbone.layers.0.norm.weight", &[HIDDEN]);
    for name in ["A_log", "D", "dt_bias"] {
        add(&mut tensors, &format!("backbone.layers.0.mixer.{name}"), &[4]);
    }
    add(&mut tensors, "backbone.layers.0.mixer.conv1d.weight", &[192, 4, 1]);
    add(&mut tensors, "backbone.layers.0.mixer.conv1d.bias", &[192]);
    add(&mut tensors, "backbone.layers.0.mixer.in_proj.weight", &[260, HIDDEN]);
    add(&mut tensors, "backbone.layers.0.mixer.norm.weight", &[64]);
    add(&mut tensors, "backbone.layers.0.mixer.out_proj.weight", &[HIDDEN, 64]);
    add(&mut tensors, "backbone.layers.1.norm.weight", &[HIDDEN]);
    for (proj, shape) in
        [("q", [HIDDEN, HIDDEN]), ("k", [16, HIDDEN]), ("v", [16, HIDDEN]), ("o", [HIDDEN, HIDDEN])]
    {
        add(&mut tensors, &format!("backbone.layers.1.mixer.{proj}_proj.weight"), &shape);
    }
    add(&mut tensors, "backbone.layers.2.norm.weight", &[HIDDEN]);
    add(&mut tensors, "backbone.layers.2.mixer.gate.weight", &[4, HIDDEN]);
    add(&mut tensors, "backbone.layers.2.mixer.gate.e_score_correction_bias", &[4]);
    add(&mut tensors, "backbone.layers.2.mixer.switch_mlp.fc1.weight", &[4, 8, HIDDEN]);
    add(&mut tensors, "backbone.layers.2.mixer.switch_mlp.fc2.weight", &[4, HIDDEN, 8]);
    add(
        &mut tensors,
        "backbone.layers.2.mixer.shared_experts.up_proj.weight",
        &[16, HIDDEN],
    );
    add(
        &mut tensors,
        "backbone.layers.2.mixer.shared_experts.down_proj.weight",
        &[HIDDEN, 16],
    );
    admitted_dense(&config, &TensorCatalog::new(tensors))
}
