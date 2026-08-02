use super::*;
use crate::ModelsError;

#[test]
fn binds_autoawq_gemm_w4a16_weight() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.qweight", "I32", vec![32, 4]),
            tensor("model.layers.0.self_attn.q_proj.qzeros", "I32", vec![4, 4]),
            tensor("model.layers.0.self_attn.q_proj.scales", "F16", vec![4, 32]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder()?, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert!(bindings.uses_awq());
    assert_eq!(bindings.tensors.len(), 1);
    let binding = &bindings.tensors[0];
    assert_eq!(binding.logical_shape.as_deref(), Some([32, 32].as_slice()));
    let TensorStorage::Awq { format, scales, zero_points } = &binding.storage else {
        return Err(ModelsError::InvalidConfig("expected AWQ storage".into()));
    };
    assert!(format.is_gemm_w4a16());
    assert_eq!(format.group_size, 8);
    assert_eq!(scales, "model.layers.0.self_attn.q_proj.scales");
    assert_eq!(zero_points, "model.layers.0.self_attn.q_proj.qzeros");
    Ok(())
}

#[test]
fn rejects_autoawq_without_packed_zero_points() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.qweight", "I32", vec![32, 4]),
            tensor("model.layers.0.self_attn.q_proj.scales", "F16", vec![4, 32]),
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
        "num_key_value_heads": 4,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))
}
