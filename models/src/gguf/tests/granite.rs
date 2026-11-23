//! Granite's scaling multipliers from a `granite` header.

use super::{TestResult, Writer};
use crate::{
    gguf::{GgmlType, GgufModel, config::config},
    layout::DecoderConfig,
};

#[test]
fn granite_scales_map_to_their_checkpoint_names() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "granite")?;
    for (key, value) in [
        ("embedding_scale", 12.0_f32),
        ("attention.scale", 0.015_625),
        ("residual_scale", 0.22),
        ("logit_scale", 8.0),
    ] {
        writer = writer.kv(&format!("granite.{key}"), 6, value.to_le_bytes().to_vec());
    }
    for (key, value) in [
        ("embedding_length", 64_u32),
        ("block_count", 1),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 2),
        ("feed_forward_length", 128),
    ] {
        writer = writer.kv(&format!("granite.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer.tensor("token_embd.weight", &[64, 8], GgmlType::Q8_0)?;
    let path =
        std::env::temp_dir().join(format!("libmir-gguf-granite-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let mut derived = derived?;
    derived["vocab_size"] = 8.into();
    assert_eq!(derived["logits_scaling"], 8.0);
    assert_eq!(derived["residual_multiplier"], 0.22);
    let decoder = DecoderConfig::from_value(&derived)?;
    assert_eq!(decoder.embedding_multiplier, Some(12.0));
    assert_eq!(decoder.attention_scale, Some(0.015_625));
    assert_eq!((decoder.residual_multiplier, decoder.logits_divisor), (Some(0.22), Some(8.0)));
    Ok(())
}
