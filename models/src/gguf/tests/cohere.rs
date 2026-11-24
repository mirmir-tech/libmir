//! Command R7B's block, norm, rotation and logit scale from a `cohere2`
//! header.

use super::{TestResult, Writer};
use crate::{
    gguf::{GgmlType, GgufModel, config::config},
    layout::{AttentionLayerType, DecoderConfig, NormKind},
};

#[test]
fn cohere2_declares_parallel_layer_norm_blocks_and_scaled_logits() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "cohere2")?;
    for (key, value) in [("logit_scale", 0.25_f32), ("attention.layer_norm_epsilon", 1.0e-5)] {
        writer = writer.kv(&format!("cohere2.{key}"), 6, value.to_le_bytes().to_vec());
    }
    for (key, value) in [
        ("embedding_length", 64_u32),
        ("block_count", 4),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 2),
        ("feed_forward_length", 128),
        ("attention.sliding_window", 16),
    ] {
        writer = writer.kv(&format!("cohere2.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer.tensor("token_embd.weight", &[64, 8], GgmlType::Q8_0)?;
    let path =
        std::env::temp_dir().join(format!("libmir-gguf-cohere2-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let mut derived = derived?;
    derived["vocab_size"] = 8.into();
    assert_eq!(derived["logit_scale"], 0.25);
    let decoder = DecoderConfig::from_value(&derived)?;
    assert_eq!(decoder.logits_divisor, Some(4.0));
    assert!(decoder.parallel_block);
    let rotates = (0..4).map(|layer| decoder.attention_pattern.rotates(layer));
    assert_eq!(rotates.collect::<Vec<_>>(), [true, true, true, false]);
    assert_eq!(decoder.norm_kind, NormKind::LayerNorm);
    assert!((decoder.rms_norm_eps - 1.0e-5).abs() < 1.0e-12);
    let full = (0..4).map(|layer| decoder.layer_type(layer) == AttentionLayerType::Full);
    assert_eq!(full.collect::<Vec<_>>(), [false, false, false, true]);
    Ok(())
}
