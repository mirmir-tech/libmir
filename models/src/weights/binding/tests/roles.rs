use super::*;

#[test]
fn rejects_a_physical_shape_that_violates_the_semantic_role() -> Result<()> {
    let decoder = dense_decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![tensor("model.layers.0.self_attn.q_proj.weight", "BF16", vec![31, 32])],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn exposes_typed_roles_with_semantic_shapes() -> Result<()> {
    let decoder = dense_decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![tensor("model.layers.0.self_attn.k_proj.weight", "BF16", vec![16, 32])],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    let role = LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::AttentionProjection { projection: AttentionProjectionRole::Key },
    };
    let Some(binding) = plan.binding(&role) else {
        return Err(crate::ModelsError::InvalidConfig("typed role is not bound".into()));
    };

    assert_eq!(binding.logical_shape.as_deref(), Some([16, 32].as_slice()));
    assert_eq!(binding.source, "model.layers.0.self_attn.k_proj.weight");
    Ok(())
}

#[test]
fn records_fused_qkv_and_transposition_as_binding_transforms() -> Result<()> {
    let decoder = dense_decoder()?;
    let fused_catalog = TensorCatalog {
        tensors: vec![tensor("model.layers.0.self_attn.qkv_proj.weight", "BF16", vec![64, 32])],
    };
    let spec = SemanticModelSpec::discover(&decoder, &fused_catalog)?;
    let fused = WeightBindingPlan::discover(&spec, &fused_catalog)?;
    assert!(fused.tensors[0].transforms.contains(&BindingTransform::FusedQkv {
        query: 32,
        key: 16,
        value: 16,
    }));

    let transposed_catalog = TensorCatalog {
        tensors: vec![tensor("model.layers.0.self_attn.k_proj.weight", "BF16", vec![32, 16])],
    };
    let transposed = WeightBindingPlan::discover(&spec, &transposed_catalog)?;
    assert!(transposed.tensors[0].transforms.contains(&BindingTransform::Transpose));
    Ok(())
}

#[test]
fn rejects_ambiguous_fused_and_separate_qkv_grammars() -> Result<()> {
    let decoder = dense_decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.qkv_proj.weight", "BF16", vec![64, 32]),
            tensor("model.layers.0.self_attn.q_proj.weight", "BF16", vec![32, 32]),
            tensor("model.layers.0.self_attn.k_proj.weight", "BF16", vec![16, 32]),
            tensor("model.layers.0.self_attn.v_proj.weight", "BF16", vec![16, 32]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn keeps_affine_parameters_separate_from_the_projection_bias() -> Result<()> {
    let decoder = dense_decoder()?;
    let mut catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight", "U32", vec![32, 4]),
            tensor("model.layers.0.self_attn.q_proj.scales", "F16", vec![32, 2]),
            tensor("model.layers.0.self_attn.q_proj.biases", "F16", vec![32, 2]),
            tensor("model.layers.0.self_attn.q_proj.bias", "BF16", vec![32]),
        ],
    };
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = WeightBindingPlan::discover(&spec, &catalog)?;

    assert_eq!(
        plan.tensors[0].physical_sources(),
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases",
            "model.layers.0.self_attn.q_proj.bias",
        ]
    );
    Ok(())
}

fn dense_decoder() -> Result<DecoderConfig> {
    DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))
}
