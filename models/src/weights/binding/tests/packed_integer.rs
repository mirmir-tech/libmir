use super::*;
use crate::ModelsError;

#[test]
fn binds_compressed_tensors_packed_int8_weight() -> Result<()> {
    let decoder = decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight_packed", "I32", vec![32, 8]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "BF16", vec![32, 1]),
            tensor("model.layers.0.self_attn.q_proj.weight_shape", "I64", vec![2]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert!(bindings.uses_packed_int8());
    assert_eq!(bindings.tensors.len(), 1);
    let binding = &bindings.tensors[0];
    assert_eq!(binding.source, "model.layers.0.self_attn.q_proj.weight_packed");
    assert_eq!(binding.shape, [32, 8]);
    assert_eq!(binding.logical_shape.as_deref(), Some([32, 32].as_slice()));
    let TensorStorage::PackedInt8 {
        format,
        scales,
        shape,
        zero_points,
        group_indices,
    } = &binding.storage
    else {
        return Err(ModelsError::InvalidConfig("expected packed INT8 storage".into()));
    };
    assert!(format.is_symmetric_channel_int8());
    assert_eq!(format.scale_dtype, CompressedIntegerScaleDType::BF16);
    assert_eq!(scales, "model.layers.0.self_attn.q_proj.weight_scale");
    assert_eq!(shape, "model.layers.0.self_attn.q_proj.weight_shape");
    assert!(zero_points.is_none());
    assert!(group_indices.is_none());
    Ok(())
}

#[test]
fn binds_symmetric_grouped_compressed_tensors_int4_weight() -> Result<()> {
    let decoder = decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight_packed", "I32", vec![32, 4]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "BF16", vec![32, 4]),
            tensor("model.layers.0.self_attn.q_proj.weight_shape", "I64", vec![2]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert!(bindings.uses_packed_int4());
    assert!(!bindings.uses_packed_int8());
    let TensorStorage::PackedInt4 {
        format,
        scales,
        zero_points,
        group_indices,
        ..
    } = &bindings.tensors[0].storage
    else {
        return Err(ModelsError::InvalidConfig("expected packed INT4 storage".into()));
    };
    assert!(format.is_symmetric_group_int4());
    assert_eq!(format.scale_strategy, CompressedIntegerScaleStrategy::Group { group_size: 8 });
    assert_eq!(scales, "model.layers.0.self_attn.q_proj.weight_scale");
    assert!(zero_points.is_none());
    assert!(group_indices.is_none());
    Ok(())
}

#[test]
fn rejects_packed_int8_without_its_shape_contract() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight_packed", "I32", vec![32, 8]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "BF16", vec![32, 1]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder()?, &catalog)?;
    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn rejects_unimplemented_asymmetric_packed_int8() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight_packed", "I32", vec![32, 8]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "BF16", vec![32, 1]),
            tensor("model.layers.0.self_attn.q_proj.weight_shape", "I64", vec![2]),
            tensor("model.layers.0.self_attn.q_proj.weight_zero_point", "I32", vec![1, 8]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder()?, &catalog)?;
    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn rejects_non_floating_packed_int8_scales() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight_packed", "I32", vec![32, 8]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "I32", vec![32, 1]),
            tensor("model.layers.0.self_attn.q_proj.weight_shape", "I64", vec![2]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder()?, &catalog)?;
    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

fn decoder() -> Result<DecoderConfig> {
    DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))
}
