//! Gemma 4 text decoders from a `gemma4` header.

use super::{TestResult, Writer};
use crate::gguf::{GgmlType, GgufModel, config::config};

fn array(kind: u32, values: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend(u64::try_from(values.len()).unwrap_or_default().to_le_bytes());
    bytes.extend(values.iter().flatten());
    bytes
}

#[test]
fn gemma4_declares_its_layer_types_and_head_widths() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "gemma4")?;
    for (key, value) in [
        ("block_count", 2_u32),
        ("attention.key_length", 512),
        ("attention.key_length_swa", 256),
        ("expert_count", 4),
        ("expert_used_count", 2),
    ] {
        writer = writer.kv(&format!("gemma4.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer
        .kv("gemma4.attention.sliding_window_pattern", 9, array(7, &[vec![1], vec![0]]))
        .kv(
            "gemma4.attention.head_count_kv",
            9,
            array(4, &[8_u32.to_le_bytes().to_vec(), 2_u32.to_le_bytes().to_vec()]),
        )
        .kv("gemma4.final_logit_softcapping", 6, 30.0_f32.to_le_bytes().to_vec())
        .tensor("token_embd.weight", &[256, 8], GgmlType::Q8_0)?
        .tensor("blk.0.attn_v.weight", &[256, 8], GgmlType::Q8_0)?;
    let path = std::env::temp_dir().join(format!("libmir-gguf-gemma4-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let derived = derived?;
    assert_eq!(derived["model_type"], "gemma4_text");
    assert_eq!(
        derived["layer_types"],
        serde_json::json!(["sliding_attention", "full_attention"])
    );
    assert_eq!(
        (derived["head_dim"].clone(), derived["global_head_dim"].clone()),
        (256.into(), 512.into())
    );
    assert_eq!(
        (
            derived["num_key_value_heads"].clone(),
            derived["num_global_key_value_heads"].clone()
        ),
        (8.into(), 2.into())
    );
    // The full layer has no value projection.
    assert_eq!(derived["attention_k_eq_v"], true);
    assert_eq!(derived["rope_parameters"]["full_attention"]["rope_type"], "proportional");
    assert_eq!(derived["final_logit_softcapping"], 30.0);
    Ok(())
}
