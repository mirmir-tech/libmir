use super::*;

#[test]
fn physical_checkpoint_formats_do_not_change_model_semantics() -> Result<()> {
    let decoder = decoder()?;
    let native = native_catalog();
    let mlx = mlx_catalog();
    let native_spec = SemanticModelSpec::discover(&decoder, &native)?;
    let mlx_spec = SemanticModelSpec::discover(&decoder, &mlx)?;

    assert_eq!(native_spec, mlx_spec);
    let native_bindings = WeightBindingPlan::discover(&native_spec, &native)?;
    let mlx_bindings = WeightBindingPlan::discover(&mlx_spec, &mlx)?;
    assert!(
        native_bindings
            .tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::BlockQuantized { .. }))
    );
    assert!(
        mlx_bindings
            .tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::AffineQuantized { .. }))
    );
    assert_eq!(
        native_bindings.expert_projection_layout(),
        Some(ExpertProjectionLayout::InterleavedGateUp)
    );
    let expert = |projection| LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
    };
    assert_eq!(
        native_bindings
            .binding(&expert(ExpertProjectionRole::GateUp))
            .and_then(TensorBinding::block_projection_layout),
        Some(BlockProjectionLayout::FusedGateUpBank { experts: 8, interleaved: true })
    );
    assert_eq!(
        native_bindings
            .binding(&expert(ExpertProjectionRole::Down))
            .and_then(TensorBinding::block_projection_layout),
        Some(BlockProjectionLayout::MatrixBank { matrices: 8 })
    );
    assert_eq!(
        mlx_bindings.expert_projection_layout(),
        Some(ExpertProjectionLayout::SeparateGateUp)
    );
    assert_ne!(native_bindings, mlx_bindings);
    Ok(())
}

#[test]
fn rejects_ambiguous_expert_binding_grammars() -> Result<()> {
    let decoder = decoder()?;
    let mut ambiguous = native_catalog();
    ambiguous.tensors.extend(mlx_catalog().tensors);
    let spec = SemanticModelSpec::discover(&decoder, &ambiguous)?;

    assert!(WeightBindingPlan::discover(&spec, &ambiguous).is_err());
    Ok(())
}

#[test]
fn rejects_nvfp4_with_a_mismatched_block_scale_dtype() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj.weight", "U8", vec![32, 16]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale", "F32", vec![32, 2]),
            tensor("model.layers.0.self_attn.q_proj.weight_scale_2", "F32", Vec::new()),
            tensor("model.layers.0.self_attn.q_proj.input_scale", "F32", Vec::new()),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn binds_compressed_tensors_nvfp4_names() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))?;
    let catalog = TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.q_proj.input_global_scale", "F32", vec![1]),
        tensor("model.layers.0.self_attn.q_proj.weight_global_scale", "F32", vec![1]),
        tensor("model.layers.0.self_attn.q_proj.weight_packed", "U8", vec![32, 16]),
        tensor("model.layers.0.self_attn.q_proj.weight_scale", "F8_E4M3", vec![32, 2]),
    ]);
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;

    assert_eq!(bindings.tensors.len(), 1);
    assert!(bindings.uses_block_format(BlockFormat::NvFp4));
    assert!(matches!(
        bindings.tensors[0].storage,
        TensorStorage::BlockQuantized {
            format: BlockQuantization::NVFP4,
            global_scale: Some(_),
            input_scale: Some(_),
            ..
        }
    ));
    Ok(())
}

#[test]
fn rejects_mxfp4_with_a_non_bf16_output_bias() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "hidden_act": "silu"
    }))?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.layers.0.self_attn.q_proj_blocks", "U8", vec![32, 1, 16]),
            tensor("model.layers.0.self_attn.q_proj_scales", "U8", vec![32, 1]),
            tensor("model.layers.0.self_attn.q_proj_bias", "F16", vec![32]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

fn native_catalog() -> TensorCatalog {
    TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.sinks", "F32", vec![4]),
        tensor("model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", vec![8, 64, 1, 16]),
        tensor("model.layers.0.mlp.experts.gate_up_proj_scales", "U8", vec![8, 64, 1]),
        tensor("model.layers.0.mlp.experts.down_proj_blocks", "U8", vec![8, 32, 1, 16]),
        tensor("model.layers.0.mlp.experts.down_proj_scales", "U8", vec![8, 32, 1]),
    ])
}

fn mlx_catalog() -> TensorCatalog {
    TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.sinks", "F32", vec![4]),
        tensor("model.layers.0.mlp.experts.gate_proj.weight", "U32", vec![8, 32, 4]),
        tensor("model.layers.0.mlp.experts.gate_proj.scales", "F16", vec![8, 32, 1]),
        tensor("model.layers.0.mlp.experts.up_proj.weight", "U32", vec![8, 32, 4]),
        tensor("model.layers.0.mlp.experts.up_proj.scales", "F16", vec![8, 32, 1]),
        tensor("model.layers.0.mlp.experts.down_proj.weight", "U32", vec![8, 32, 4]),
        tensor("model.layers.0.mlp.experts.down_proj.scales", "F16", vec![8, 32, 1]),
    ])
}
