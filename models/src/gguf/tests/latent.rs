//! Latent-attention routing a `deepseek2` header declares.

use super::{TestResult, Writer};
use crate::gguf::{GgmlType, GgufModel, config::config};

fn routing(gating: Option<u32>) -> TestResult<serde_json::Value> {
    let unsigned = |value: u32| value.to_le_bytes().to_vec();
    let mut writer = Writer::new()
        .text("general.architecture", "deepseek2")?
        .kv("deepseek2.attention.kv_lora_rank", 4, unsigned(512))
        .kv("deepseek2.expert_group_count", 4, unsigned(8))
        .kv("deepseek2.expert_group_used_count", 4, unsigned(3));
    if let Some(gating) = gating {
        writer = writer.kv("deepseek2.expert_gating_func", 4, unsigned(gating));
    }
    let writer = writer.tensor("token_embd.weight", &[256, 8], GgmlType::Q4_K)?;
    let path = std::env::temp_dir()
        .join(format!("libmir-gguf-latent-{}-{gating:?}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    Ok(derived?)
}

#[test]
fn grouped_routers_keep_their_group_ranking() -> TestResult {
    let field = |config: &serde_json::Value, key: &str| config[key].clone();
    // A softmax router over several groups is `group_limited_greedy`.
    let softmax = routing(None)?;
    assert_eq!(field(&softmax, "topk_method"), "group_limited_greedy");
    assert_eq!(
        (field(&softmax, "n_group"), field(&softmax, "topk_group")),
        (8.into(), 3.into())
    );
    // llama.cpp's sigmoid gating is `noaux_tc`.
    let sigmoid = routing(Some(2))?;
    assert_eq!(field(&sigmoid, "topk_method"), "noaux_tc");
    assert_eq!(field(&sigmoid, "scoring_func"), "sigmoid");
    Ok(())
}
