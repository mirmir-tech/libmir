use super::*;

#[test]
fn admits_dense_clamped_routed_generation_on_metal_and_cuda() -> Result<()> {
    let config = json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_act": "silu",
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "layer_types": ["full_attention"],
        "rope_theta": 150_000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 32
        },
        "tie_word_embeddings": false
    });
    let contract = RemoteModelContract::inspect_generation(&config, &catalog())?;
    let metal = contract.admission(BackendTarget::Metal);
    let cuda = contract.admission(BackendTarget::Cuda);

    for report in [&metal, &cuda] {
        assert_eq!(report.status, AdmissionStatus::Supported);
        assert!(report.checks.iter().any(|check| {
            check.kind == AdmissionCheckKind::Dense && check.status == AdmissionStatus::Supported
        }));
    }
    Ok(())
}

fn catalog() -> TensorCatalog {
    let layer = "model.layers.0";
    let specs = [
        ("model.embed_tokens.weight".into(), vec![64, 32]),
        ("model.norm.weight".into(), vec![32]),
        ("lm_head.weight".into(), vec![64, 32]),
        (format!("{layer}.input_layernorm.weight"), vec![32]),
        (format!("{layer}.self_attn.q_proj.weight"), vec![32, 32]),
        (format!("{layer}.self_attn.q_proj.bias"), vec![32]),
        (format!("{layer}.self_attn.k_proj.weight"), vec![16, 32]),
        (format!("{layer}.self_attn.k_proj.bias"), vec![16]),
        (format!("{layer}.self_attn.v_proj.weight"), vec![16, 32]),
        (format!("{layer}.self_attn.v_proj.bias"), vec![16]),
        (format!("{layer}.self_attn.o_proj.weight"), vec![32, 32]),
        (format!("{layer}.self_attn.o_proj.bias"), vec![32]),
        (format!("{layer}.self_attn.sinks"), vec![4]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![32]),
        (format!("{layer}.mlp.router.weight"), vec![2, 32]),
        (format!("{layer}.mlp.experts.gate_proj.weight"), vec![2, 32, 32]),
        (format!("{layer}.mlp.experts.up_proj.weight"), vec![2, 32, 32]),
        (format!("{layer}.mlp.experts.down_proj.weight"), vec![2, 32, 32]),
    ];
    TensorCatalog::new(
        specs
            .into_iter()
            .map(|(name, shape)| tensor_with_dtype(&name, shape, "F32"))
            .collect(),
    )
}
