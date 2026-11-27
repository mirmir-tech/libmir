//! Gated `DeltaNet` hybrids a `qwen35` header declares.

use super::{TestResult, Writer, parse};
use crate::gguf::{GgmlType, GgufModel, config::config};

fn unsigned(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn sections(values: &[u32]) -> Vec<u8> {
    let mut bytes = unsigned(4);
    bytes.extend(u64::try_from(values.len()).unwrap_or_default().to_le_bytes());
    bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
    bytes
}

fn derive() -> TestResult<serde_json::Value> {
    let mut writer = Writer::new().text("general.architecture", "qwen35")?;
    for (key, value) in [
        // Eight decoder layers and one multi-token prediction layer.
        ("block_count", 9),
        ("nextn_predict_layers", 1),
        ("full_attention_interval", 4),
        ("attention.key_length", 256),
        ("rope.dimension_count", 64),
        ("ssm.conv_kernel", 4),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 32),
        ("ssm.state_size", 128),
        ("ssm.inner_size", 4096),
    ] {
        writer = writer.kv(&format!("qwen35.{key}"), 4, unsigned(value));
    }
    let writer = writer
        .kv("qwen35.rope.dimension_sections", 9, sections(&[11, 11, 10, 0]))
        .tensor("token_embd.weight", &[256, 8], GgmlType::Q4_K)?;
    let path = std::env::temp_dir().join(format!("libmir-gguf-hybrid-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    Ok(derived?)
}

#[test]
fn qwen35_declares_its_linear_attention_layout() -> TestResult {
    let config = derive()?;
    assert_eq!(config["model_type"], "qwen3_5_text");
    assert_eq!(
        (config["num_hidden_layers"].clone(), config["num_nextn_predict_layers"].clone()),
        (8.into(), 1.into())
    );
    assert_eq!(config["linear_num_key_heads"], 16);
    assert_eq!(config["linear_num_value_heads"], 32);
    assert_eq!(config["linear_value_head_dim"], 128);
    assert_eq!(config["mir_linear_attention_layout"], "llama_cpp");
    let layers = config["layer_types"].as_array().ok_or("layer_types is not a list")?;
    let full = layers.iter().filter(|layer| *layer == "full_attention").count();
    assert_eq!((layers.len(), full, &layers[3]), (8, 2, &"full_attention".into()));
    assert_eq!(config["partial_rotary_factor"], 0.25);
    assert_eq!(config["rope_parameters"]["mrope_section"], serde_json::json!([11, 11, 10]));
    assert_eq!(config["rope_parameters"]["mrope_interleaved"], true);
    Ok(())
}

#[test]
fn shared_expert_gate_keeps_its_unit_output_axis() -> TestResult {
    let bytes = Writer::new()
        .text("general.architecture", "qwen35moe")?
        .tensor("blk.0.ffn_gate_inp_shexp.weight", &[8], GgmlType::F32)?
        .bytes()?;
    let header = parse(&bytes).map_err(|error| format!("{error:?}"))?;
    let catalog = header.mapped_catalog(std::path::Path::new("model.gguf"))?;
    let gate = catalog
        .get("model.layers.0.mlp.shared_expert_gate.weight")
        .ok_or("gate is missing")?;
    assert_eq!(gate.shape, [1, 8]);
    Ok(())
}
