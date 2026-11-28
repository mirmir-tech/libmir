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
fn rejects_a_dense_bias_with_a_different_dtype() -> Result<()> {
    let decoder = dense_decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight", "BF16", vec![32, 32]),
            tensor("model.layers.0.self_attn.q_proj.bias", "F16", vec![32]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn admits_an_f32_gguf_bias_beside_half_precision_weights() -> Result<()> {
    // GGUF keeps every bias in F32 beside F16 and BF16 projections, and its
    // loaders convert both to the activation dtype.
    let decoder = dense_decoder()?;
    for weight in ["BF16", "F16"] {
        let catalog = biased_catalog(weight, crate::layout::WeightContainer::Gguf, "model.gguf");
        let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
        let plan = WeightBindingPlan::discover(&spec, &catalog)?;
        assert_eq!(plan.tensors[0].dense_bias(), Some("model.layers.0.self_attn.q_proj.bias"));
    }
    Ok(())
}

#[test]
fn rejects_an_f32_safetensors_bias_beside_half_precision_weights() -> Result<()> {
    // SafeTensors biases load as stored, so an F32 bias would widen the
    // projection's output.
    let decoder = dense_decoder()?;
    let catalog =
        biased_catalog("BF16", crate::layout::WeightContainer::SafeTensors, "model.safetensors");
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let error = WeightBindingPlan::discover(&spec, &catalog)
        .err()
        .map(|error| error.to_string());
    assert!(
        error
            .as_deref()
            .is_some_and(|error| error.contains("expected dtype BF16, found F32")),
        "{error:?}"
    );
    Ok(())
}

fn biased_catalog(
    weight: &str,
    container: crate::layout::WeightContainer,
    file: &str,
) -> TensorCatalog {
    let mut tensors = vec![
        tensor("model.layers.0.self_attn.q_proj.bias", "F32", vec![32]),
        tensor("model.layers.0.self_attn.q_proj.weight", weight, vec![32, 32]),
    ];
    for tensor in &mut tensors {
        tensor.container = container;
        tensor.file = file.into();
    }
    TensorCatalog { tensors }
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
