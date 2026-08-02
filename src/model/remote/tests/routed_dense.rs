use super::*;

#[test]
fn admits_stacked_dense_and_routed_generation_on_metal_and_cuda() -> Result<()> {
    let config = json!({
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 8,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 4,
        "hidden_act": "gelu_pytorch_tanh",
        "attention_k_eq_v": true,
        "tie_word_embeddings": true
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
    let layer = "language_model.model.layers.0";
    let specs = [
        ("language_model.model.embed_tokens.weight".into(), vec![8, 4]),
        ("language_model.model.norm.weight".into(), vec![4]),
        (format!("{layer}.input_layernorm.weight"), vec![4]),
        (format!("{layer}.self_attn.q_proj.weight"), vec![4, 4]),
        (format!("{layer}.self_attn.k_proj.weight"), vec![4, 4]),
        (format!("{layer}.self_attn.o_proj.weight"), vec![4, 4]),
        (format!("{layer}.self_attn.q_norm.weight"), vec![4]),
        (format!("{layer}.self_attn.k_norm.weight"), vec![4]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![4]),
        (format!("{layer}.pre_feedforward_layernorm.weight"), vec![4]),
        (format!("{layer}.mlp.gate_proj.weight"), vec![8, 4]),
        (format!("{layer}.mlp.up_proj.weight"), vec![8, 4]),
        (format!("{layer}.mlp.down_proj.weight"), vec![4, 8]),
        (format!("{layer}.post_feedforward_layernorm_1.weight"), vec![4]),
        (format!("{layer}.router.proj.weight"), vec![2, 4]),
        (format!("{layer}.router.scale"), vec![4]),
        (format!("{layer}.router.per_expert_scale"), vec![2]),
        (format!("{layer}.pre_feedforward_layernorm_2.weight"), vec![4]),
        (format!("{layer}.experts.switch_glu.gate_proj.weight"), vec![2, 4, 4]),
        (format!("{layer}.experts.switch_glu.up_proj.weight"), vec![2, 4, 4]),
        (format!("{layer}.experts.switch_glu.down_proj.weight"), vec![2, 4, 4]),
        (format!("{layer}.post_feedforward_layernorm_2.weight"), vec![4]),
        (format!("{layer}.post_feedforward_layernorm.weight"), vec![4]),
        (format!("{layer}.layer_scalar"), vec![1]),
    ];
    TensorCatalog {
        tensors: specs
            .into_iter()
            .map(|(name, shape)| tensor_with_dtype(&name, shape, "F32"))
            .collect(),
    }
}
