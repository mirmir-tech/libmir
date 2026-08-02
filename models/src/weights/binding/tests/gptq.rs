use std::fs;

use super::*;
use crate::{ModelsError, layout::ModelLayout};

#[test]
fn binds_configured_gptq_input_packing() -> Result<()> {
    for (checkpoint, symmetric, expected) in [
        ("gptq", true, GptqCheckpointFormat::Gptq),
        ("gptq_v2", false, GptqCheckpointFormat::GptqV2),
    ] {
        let root = fixture(checkpoint, symmetric, "int32")?;
        let layout = ModelLayout::inspect(&root)?;
        let catalog = catalog();
        let spec = SemanticModelSpec::discover(&decoder()?, &catalog)?;
        let bindings = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
        assert!(bindings.uses_gptq());
        let TensorStorage::Gptq {
            format,
            scales,
            zero_points,
            group_indices,
        } = &bindings.tensors[0].storage
        else {
            return Err(ModelsError::InvalidConfig("expected GPTQ storage".into()));
        };
        assert_eq!(format.bits, GptqBits::Four);
        assert_eq!(format.group_size, 8);
        assert_eq!(format.checkpoint_format, expected);
        assert_eq!(format.symmetric, symmetric);
        assert!(format.activation_order);
        assert_eq!(scales, "model.layers.0.self_attn.q_proj.scales");
        assert_eq!(zero_points, "model.layers.0.self_attn.q_proj.qzeros");
        assert_eq!(group_indices, "model.layers.0.self_attn.q_proj.g_idx");
        fs::remove_dir_all(root)?;
    }
    Ok(())
}

#[test]
fn rejects_gptq_non_i32_checkpoint_packing() -> Result<()> {
    let root = fixture("gptq", true, "int8")?;
    let layout = ModelLayout::inspect(&root)?;
    let catalog = catalog();
    let spec = SemanticModelSpec::discover(&decoder()?, &catalog)?;
    assert!(WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

fn catalog() -> TensorCatalog {
    TensorCatalog::new(vec![
        tensor("model.layers.0.self_attn.q_proj.qweight", "I32", vec![4, 32]),
        tensor("model.layers.0.self_attn.q_proj.qzeros", "I32", vec![4, 4]),
        tensor("model.layers.0.self_attn.q_proj.scales", "F16", vec![4, 32]),
        tensor("model.layers.0.self_attn.q_proj.g_idx", "I32", vec![32]),
    ])
}

fn fixture(checkpoint: &str, symmetric: bool, pack: &str) -> Result<PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "libmir-gptq-binding-{}-{checkpoint}-{symmetric}-{pack}",
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({
            "quantization_config": {
                "bits": 4,
                "checkpoint_format": checkpoint,
                "desc_act": true,
                "group_size": 8,
                "pack_dtype": pack,
                "quant_method": "gptq",
                "sym": symmetric
            }
        }))?,
    )?;
    fs::write(root.join("model.safetensors"), [])?;
    Ok(root)
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
