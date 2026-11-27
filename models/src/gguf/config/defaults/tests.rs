use serde_json::{Map, Value, json};

use super::fixed;

fn gemma3(layers: u64, hidden: u64, heads: u64) -> Map<String, Value> {
    let mut config = Map::new();
    config.insert("num_hidden_layers".into(), json!(layers));
    config.insert("hidden_size".into(), json!(hidden));
    config.insert("num_attention_heads".into(), json!(heads));
    config.insert("head_dim".into(), json!(128));
    fixed("gemma3", &mut config);
    config
}

#[test]
fn gemma3_27b_scales_attention_by_hidden_per_head() {
    // google/gemma-3-27b-it: 62 layers, 5376 / 32 = 168.
    assert_eq!(gemma3(62, 5376, 32)["query_pre_attn_scalar"], json!(168));
    // Every other size scales by its head dimension.
    assert_eq!(gemma3(48, 3840, 16)["query_pre_attn_scalar"], json!(128));
}

#[test]
fn imports_the_architectures_it_knows_and_nemotron_h() {
    for architecture in
        ["llama", "qwen2moe", "qwen3moe", "gemma4", "gpt-oss", "deepseek2", "nemotron_h"]
    {
        assert!(super::imports(architecture), "{architecture}");
    }
    for architecture in ["bert", "mamba", "t5", ""] {
        assert!(!super::imports(architecture), "{architecture}");
    }
}

/// llama.cpp renormalizes Qwen3-MoE's selected experts, which transformers'
/// configuration does not by default: the import declares it, unless the
/// configuration already does.
#[test]
fn qwen3_moe_declares_its_renormalized_experts() {
    let mut config = Map::new();
    fixed("qwen3moe", &mut config);
    assert_eq!(config["norm_topk_prob"], json!(true));
    let mut declared = Map::new();
    declared.insert("norm_topk_prob".into(), json!(false));
    fixed("qwen3moe", &mut declared);
    assert_eq!(declared["norm_topk_prob"], json!(false));
    let mut other = Map::new();
    fixed("qwen3", &mut other);
    assert!(!other.contains_key("norm_topk_prob"));
}
