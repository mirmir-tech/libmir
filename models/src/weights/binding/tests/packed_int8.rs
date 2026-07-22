use super::*;

#[test]
fn binds_compressed_tensors_packed_int8_weight() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))?;
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
    assert!(matches!(
        &binding.storage,
        TensorStorage::PackedInt8 { dtype, scales }
            if dtype == "I32"
                && scales == "model.layers.0.self_attn.q_proj.weight_scale"
    ));
    Ok(())
}
