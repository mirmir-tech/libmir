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
