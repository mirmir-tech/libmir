//! GLM-4.5's expert layout from a `glm4moe` header: llama.cpp gates it by
//! sigmoid when `expert_gating_func` is absent.

use super::{TestResult, Writer};
use crate::{
    gguf::{GgmlType, GgufModel, config::config},
    layout::{DecoderConfig, ExpertRouting},
};

#[test]
fn glm4moe_declares_the_deepseek_expert_layout() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "glm4moe")?;
    for (key, value) in [
        ("embedding_length", 64_u32),
        ("block_count", 3),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 16),
        ("attention.value_length", 16),
        ("rope.dimension_count", 8),
        ("feed_forward_length", 160),
        ("expert_count", 8),
        ("expert_used_count", 2),
        ("expert_shared_count", 1),
        ("expert_feed_forward_length", 32),
        ("leading_dense_block_count", 1),
    ] {
        writer = writer.kv(&format!("glm4moe.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer.kv("glm4moe.expert_weights_norm", 7, vec![1]).tensor(
        "token_embd.weight",
        &[64, 8],
        GgmlType::Q8_0,
    )?;
    let path =
        std::env::temp_dir().join(format!("libmir-gguf-glm4moe-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let mut derived = derived?;
    derived["vocab_size"] = 8.into();
    assert_eq!(derived["model_type"], "glm4_moe");
    assert_eq!(derived["scoring_func"], "sigmoid");
    assert_eq!(derived["partial_rotary_factor"], 0.5);
    let decoder = DecoderConfig::from_value(&derived)?;
    let experts = decoder.routed_experts.ok_or("missing experts")?;
    assert_eq!(experts.routing, ExpertRouting::SigmoidWithCorrectionBias);
    assert_eq!(experts.first_dense_layers, 1);
    Ok(())
}
