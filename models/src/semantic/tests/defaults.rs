use super::*;

#[test]
fn gemma4_defaults_cover_nested_and_unified_checkpoints() -> Result<()> {
    for model_type in ["gemma4", "gemma4_text", "gemma4_unified"] {
        let mut config = configuration(model_type, &[]);
        config["hidden_size"] = json!(3840);
        config["head_dim"] = json!(256);
        config["global_head_dim"] = json!(512);
        for value in [config.clone(), json!({"model_type": "gemma4", "text_config": config})] {
            let decoder = DecoderConfig::from_value(&value)?;
            let spec = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: vec![] })?;
            assert!((spec.decoder.embedding_multiplier - 61.967_733_539_318_67).abs() < 1e-12);
            for (layer, dimension) in spec.decoder.layers.iter().zip([256, 512]) {
                let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
                    return Err(crate::error::ModelsError::InvalidConfig(
                        "expected softmax".into(),
                    ));
                };
                assert_eq!(attention.qk_normalization, QkNormalization::QueryKeyValueRms);
                assert_eq!(attention.head_dim, dimension);
                assert!((attention.scale - 1.0).abs() < f64::EPSILON);
            }
        }
    }
    Ok(())
}

#[test]
fn explicit_configuration_and_sidecar_override_builtin_defaults() -> Result<()> {
    let mut config = configuration("gemma4_unified", &[]);
    config["attention_scale"] = json!(0.25);
    config["embedding_multiplier"] = json!(2.0);
    config["qk_normalization"] = json!("query_key_rms");
    let decoder = DecoderConfig::from_value(&config)?;
    let catalog = TensorCatalog { tensors: vec![] };
    let derived = SemanticModelSpec::discover(&decoder, &catalog)?;
    assert!((derived.decoder.embedding_multiplier - 2.0).abs() < f64::EPSILON);
    let MixerSpec::SoftmaxAttention(attention) = &derived.decoder.layers[0].mixer else {
        return Err(crate::error::ModelsError::InvalidConfig("expected softmax".into()));
    };
    assert_eq!(attention.qk_normalization, QkNormalization::QueryKeyRms);
    assert!((attention.scale - 0.25).abs() < f64::EPSILON);

    let root = temp_dir("gemma-overrides");
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;
    let mut external = derived;
    external.decoder.embedding_multiplier = 3.0;
    fs::write(root.join("mir-model-spec.toml"), external.to_toml()?)?;
    let layout = ModelLayout::inspect(&root)?;
    assert_eq!(SemanticModelSpec::from_layout(&layout, &decoder, &catalog)?, external);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn invalid_explicit_embedding_multiplier_is_not_replaced_by_defaults() -> Result<()> {
    let mut config = configuration("gemma4_unified", &[]);
    config["embedding_multiplier"] = json!(-1.0);
    let decoder = DecoderConfig::from_value(&config)?;
    assert!(SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: vec![] }).is_err());
    Ok(())
}
