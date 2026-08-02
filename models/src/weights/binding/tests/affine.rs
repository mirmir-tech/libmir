use super::*;

#[test]
fn discovers_every_native_mlx_bit_width() -> Result<()> {
    for bits in [2_usize, 3, 4, 5, 6, 8] {
        let catalog = affine_catalog(&[(AttentionProjectionRole::Query, bits, "F16", "F16")]);
        let spec = SemanticModelSpec::discover(&test_decoder()?, &catalog)?;
        let bindings = WeightBindingPlan::discover(&spec, &catalog)?;
        let binding = bindings
            .binding(&attention(AttentionProjectionRole::Query))
            .ok_or_else(|| crate::ModelsError::InvalidConfig("missing query binding".into()))?;

        let TensorStorage::AffineQuantized { format, .. } = binding.storage else {
            return Err(crate::ModelsError::InvalidConfig("query is not grouped affine".into()));
        };
        assert_eq!(usize::from(format.bits.get()), bits);
        assert_eq!(format.group_size, 32);
        assert_eq!(format.scale_dtype, AffineParameterDType::F16);
        assert_eq!(format.bias_dtype, Some(AffineParameterDType::F16));
    }
    Ok(())
}

#[test]
fn preserves_mixed_bit_formats_per_binding() -> Result<()> {
    let catalog = affine_catalog(&[
        (AttentionProjectionRole::Query, 3, "BF16", "BF16"),
        (AttentionProjectionRole::Key, 6, "BF16", "BF16"),
    ]);
    let spec = SemanticModelSpec::discover(&test_decoder()?, &catalog)?;
    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert_eq!(bits(&bindings, AttentionProjectionRole::Query)?, AffineBits::Three);
    assert_eq!(bits(&bindings, AttentionProjectionRole::Key)?, AffineBits::Six);
    assert_eq!(bindings.affine_group_size(), Some(32));
    Ok(())
}

#[test]
fn rejects_unknown_width_and_mixed_parameter_dtypes() -> Result<()> {
    let malformed_width = affine_catalog(&[(AttentionProjectionRole::Query, 7, "F16", "F16")]);
    let spec = SemanticModelSpec::discover(&test_decoder()?, &malformed_width)?;
    assert!(WeightBindingPlan::discover(&spec, &malformed_width).is_err());

    let mixed_dtype = affine_catalog(&[(AttentionProjectionRole::Query, 4, "F16", "BF16")]);
    let spec = SemanticModelSpec::discover(&test_decoder()?, &mixed_dtype)?;
    assert!(WeightBindingPlan::discover(&spec, &mixed_dtype).is_err());
    Ok(())
}

#[test]
fn rejects_non_integral_packing_and_group_geometry() -> Result<()> {
    let mut packing = affine_catalog(&[(AttentionProjectionRole::Query, 4, "F16", "F16")]);
    packing.tensors[0].dtype = "U8".into();
    packing.tensors[0].shape[1] = 5;
    let spec = SemanticModelSpec::discover(&test_decoder()?, &packing)?;
    assert!(WeightBindingPlan::discover(&spec, &packing).is_err());

    let mut grouping = affine_catalog(&[(AttentionProjectionRole::Query, 4, "F16", "F16")]);
    grouping.tensors[1].shape[1] = 3;
    grouping.tensors[2].shape[1] = 3;
    let spec = SemanticModelSpec::discover(&test_decoder()?, &grouping)?;
    assert!(WeightBindingPlan::discover(&spec, &grouping).is_err());
    Ok(())
}

fn bits(bindings: &WeightBindingPlan, projection: AttentionProjectionRole) -> Result<AffineBits> {
    let binding = bindings
        .binding(&attention(projection))
        .ok_or_else(|| crate::ModelsError::InvalidConfig("missing attention binding".into()))?;
    match binding.storage {
        TensorStorage::AffineQuantized { format, .. } => Ok(format.bits),
        _ => Err(crate::ModelsError::InvalidConfig(
            "attention binding is not grouped affine".into(),
        )),
    }
}

fn attention(projection: AttentionProjectionRole) -> LogicalTensorRole {
    LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::AttentionProjection { projection },
    }
}

fn test_decoder() -> Result<DecoderConfig> {
    DecoderConfig::from_value(&serde_json::json!({
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

fn affine_catalog(formats: &[(AttentionProjectionRole, usize, &str, &str)]) -> TensorCatalog {
    let mut tensors = Vec::new();
    for (projection, bits, scale_dtype, bias_dtype) in formats {
        let name = match projection {
            AttentionProjectionRole::Query => "q_proj",
            AttentionProjectionRole::Key => "k_proj",
            AttentionProjectionRole::Value => "v_proj",
            AttentionProjectionRole::Qkv => "qkv_proj",
            AttentionProjectionRole::Output => "o_proj",
        };
        let output = match projection {
            AttentionProjectionRole::Query | AttentionProjectionRole::Output => 32,
            AttentionProjectionRole::Key | AttentionProjectionRole::Value => 16,
            AttentionProjectionRole::Qkv => 64,
        };
        let prefix = format!("model.layers.0.self_attn.{name}");
        tensors.push(tensor(&format!("{prefix}.weight"), "U32", vec![output, *bits]));
        tensors.push(tensor(&format!("{prefix}.scales"), scale_dtype, vec![output, 1]));
        tensors.push(tensor(&format!("{prefix}.biases"), bias_dtype, vec![output, 1]));
    }
    TensorCatalog::new(tensors)
}
