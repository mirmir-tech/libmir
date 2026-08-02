use super::*;

#[test]
fn admits_complete_remote_spatial_vision_contract() -> Result<()> {
    let config = spatial_vision_config();
    let processor = json!({
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 65_536,
        "max_pixels": 16_777_216,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5]
    });
    let mut catalog = spatial_vision_catalog();
    for tensor in &mut catalog.tensors {
        if tensor.name.starts_with("model.visual.") {
            tensor.dtype = "F16".into();
        }
    }
    let contract = RemoteModelContract::inspect(
        &config,
        &catalog,
        RemoteTaskMetadata {
            processor: Some(&processor),
            ..Default::default()
        },
    )?
    .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("task missing".into())))?;

    let vision = contract
        .vision()
        .ok_or_else(|| Error::Model(models::ModelsError::InvalidConfig("vision missing".into())))?;
    assert!(vision.readiness().is_ready());
    assert_eq!(vision.readiness().dtypes, ["F16"]);
    assert!(vision.processor().is_some());
    assert_eq!(contract.checkpoint_encoding().label(), "Dense BF16 + Dense F16");
    assert!(contract.admission(BackendTarget::Metal).checks.iter().any(|check| {
        check.kind == AdmissionCheckKind::Vision && check.status == AdmissionStatus::Supported
    }));
    assert_eq!(contract.admission(BackendTarget::Cuda).status, AdmissionStatus::Supported);
    Ok(())
}

#[test]
fn rejects_remote_vision_without_processor() -> Result<()> {
    let contract = RemoteModelContract::inspect_generation(
        &spatial_vision_config(),
        &spatial_vision_catalog(),
    )?;

    assert!(contract.admission(BackendTarget::Cuda).checks.iter().any(|check| {
        check.kind == AdmissionCheckKind::Vision && check.status == AdmissionStatus::Unsupported
    }));
    Ok(())
}

fn spatial_vision_config() -> serde_json::Value {
    json!({
        "architectures": ["VisionForConditionalGeneration"],
        "image_token_id": 248_056,
        "vision_start_token_id": 248_053,
        "vision_end_token_id": 248_054,
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "hidden_act": "silu",
            "rope_parameters": {"mrope_interleaved": true, "mrope_section": [4, 2, 2]}
        },
        "vision_config": {
            "depth": 0,
            "hidden_size": 32,
            "out_hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 64,
            "hidden_act": "gelu_pytorch_tanh"
        }
    })
}

fn spatial_vision_catalog() -> TensorCatalog {
    let mut catalog = dense_catalog();
    catalog.tensors.extend(
        [
            "model.visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias",
            "model.visual.pos_embed.weight",
            "model.visual.merger.norm.weight",
            "model.visual.merger.norm.bias",
            "model.visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.bias",
        ]
        .into_iter()
        .map(|name| tensor(name, Vec::new())),
    );
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    catalog
}
