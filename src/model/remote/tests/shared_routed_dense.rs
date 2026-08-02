use super::*;

#[test]
fn admits_dense_shared_routed_generation_on_metal_and_cuda() -> Result<()> {
    let config = json!({
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "attn_output_gate": true,
        "layer_types": ["linear_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "tie_word_embeddings": false
    });
    let contract = RemoteModelContract::inspect_generation(&config, &catalog())?;
    let metal = contract.admission(BackendTarget::Metal);
    let cuda = contract.admission(BackendTarget::Cuda);

    assert_eq!(metal.status, AdmissionStatus::Supported);
    for report in [&metal, &cuda] {
        assert!(report.checks.iter().any(|check| {
            check.kind == AdmissionCheckKind::Dense && check.status == AdmissionStatus::Supported
        }));
    }
    Ok(())
}

fn catalog() -> TensorCatalog {
    let layer = "language_model.model.layers.0";
    let specs = [
        ("language_model.model.embed_tokens.weight".into(), vec![64, 32]),
        ("language_model.model.norm.weight".into(), vec![32]),
        ("language_model.lm_head.weight".into(), vec![64, 32]),
        (format!("{layer}.input_layernorm.weight"), vec![32]),
        (format!("{layer}.post_attention_layernorm.weight"), vec![32]),
        (format!("{layer}.mlp.gate.weight"), vec![8, 32]),
        (format!("{layer}.mlp.switch_mlp.gate_proj.weight"), vec![8, 16, 32]),
        (format!("{layer}.mlp.switch_mlp.up_proj.weight"), vec![8, 16, 32]),
        (format!("{layer}.mlp.switch_mlp.down_proj.weight"), vec![8, 32, 16]),
        (format!("{layer}.mlp.shared_expert.gate_proj.weight"), vec![16, 32]),
        (format!("{layer}.mlp.shared_expert.up_proj.weight"), vec![16, 32]),
        (format!("{layer}.mlp.shared_expert.down_proj.weight"), vec![32, 16]),
        (format!("{layer}.mlp.shared_expert_gate.weight"), vec![1, 32]),
        (format!("{layer}.linear_attn.in_proj_qkv.weight"), vec![96, 32]),
        (format!("{layer}.linear_attn.in_proj_z.weight"), vec![32, 32]),
        (format!("{layer}.linear_attn.in_proj_a.weight"), vec![1, 32]),
        (format!("{layer}.linear_attn.in_proj_b.weight"), vec![1, 32]),
        (format!("{layer}.linear_attn.out_proj.weight"), vec![32, 32]),
        (format!("{layer}.linear_attn.conv1d.weight"), vec![96, 4, 1]),
        (format!("{layer}.linear_attn.norm.weight"), vec![32]),
        (format!("{layer}.linear_attn.A_log"), vec![1]),
        (format!("{layer}.linear_attn.dt_bias"), vec![1]),
    ];
    TensorCatalog {
        tensors: specs
            .into_iter()
            .map(|(name, shape)| tensor_with_dtype(&name, shape, "F32"))
            .collect(),
    }
}
