//! Latent-attention routing a `deepseek2` header declares.

use super::{TestResult, Writer};
use crate::gguf::{GgmlType, GgufModel, config::config};

fn unsigned(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// The config of a `deepseek2` header with `extra` keys.
fn derive(
    label: &str,
    extra: impl FnOnce(Writer) -> TestResult<Writer>,
) -> TestResult<serde_json::Value> {
    let writer = Writer::new().text("general.architecture", "deepseek2")?.kv(
        "deepseek2.attention.kv_lora_rank",
        4,
        unsigned(512),
    );
    let writer = extra(writer)?.tensor("token_embd.weight", &[256, 8], GgmlType::Q4_K)?;
    let path = std::env::temp_dir()
        .join(format!("libmir-gguf-latent-{}-{label}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    Ok(derived?)
}

fn routing(gating: Option<u32>) -> TestResult<serde_json::Value> {
    derive(&format!("{gating:?}"), |writer| {
        let writer = writer.kv("deepseek2.expert_group_count", 4, unsigned(8)).kv(
            "deepseek2.expert_group_used_count",
            4,
            unsigned(3),
        );
        Ok(match gating {
            Some(gating) => writer.kv("deepseek2.expert_gating_func", 4, unsigned(gating)),
            None => writer,
        })
    })
}

/// llama.cpp's attention temperature over the `YaRN` window.
fn temperature(length: u32) -> TestResult<serde_json::Value> {
    derive(&format!("temperature-{length}"), |writer| {
        Ok(writer
            .text("deepseek2.rope.scaling.type", "yarn")?
            .kv("deepseek2.rope.scaling.factor", 6, 128.0_f32.to_le_bytes().to_vec())
            .kv("deepseek2.rope.scaling.original_context_length", 4, unsigned(8192))
            .kv("deepseek2.attention.temperature_scale", 6, 0.1_f32.to_le_bytes().to_vec())
            .kv("deepseek2.attention.temperature_length", 4, unsigned(length)))
    })
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

#[test]
fn attention_temperature_scales_queries_past_the_yarn_window() -> TestResult {
    let config = temperature(8192)?;
    assert_eq!(config["rope_scaling"]["llama_4_scaling_beta"], 0.1);
    // A temperature window other than YaRN's has no config form.
    assert!(temperature(4096).is_err());
    Ok(())
}
