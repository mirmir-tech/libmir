use std::fs;

use super::*;
use crate::{ModelsError, layout::ModelLayout};

#[test]
fn binds_root_scoped_affine_pooled_vision_projection() -> Result<()> {
    let root = fixture()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_value(&decoder_json())?;
    let catalog = TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.q_proj.weight", "BF16", vec![32, 64]),
        tensor("embed_vision.embedding_projection.weight", "U32", vec![64, 8]),
        tensor("embed_vision.embedding_projection.scales", "BF16", vec![64, 1]),
        tensor("embed_vision.embedding_projection.biases", "BF16", vec![64, 1]),
    ]);
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
    let binding = plan
        .binding(&LogicalTensorRole::VisionProjection)
        .ok_or_else(|| ModelsError::InvalidConfig("vision projection is not bound".into()))?;

    assert_eq!(binding.logical_shape.as_deref(), Some([64, 32].as_slice()));
    assert!(matches!(
        binding.storage,
        TensorStorage::AffineQuantized { format, .. }
            if format.bits == AffineBits::Eight && format.group_size == 32
    ));
    fs::remove_dir_all(root)?;
    Ok(())
}

fn fixture() -> Result<PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "libmir-vision-binding-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({
            "text_config": decoder_json(),
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "patch_size": 2,
                "pooling_kernel_size": 2,
                "position_embedding_size": 16,
                "rms_norm_eps": 1.0e-6,
                "hidden_activation": "gelu_pytorch_tanh"
            },
            "image_token_id": 10,
            "boi_token_id": 11,
            "eoi_token_id": 12,
            "vision_soft_tokens_per_image": 4,
            "quantization": {"group_size": 32, "bits": 8, "mode": "affine"}
        }))?,
    )?;
    fs::write(root.join("model.safetensors"), [])?;
    Ok(root)
}

fn decoder_json() -> serde_json::Value {
    json!({
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "vocab_size": 128,
        "hidden_act": "silu"
    })
}
