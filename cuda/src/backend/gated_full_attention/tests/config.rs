use models::layout::DecoderConfig;
use serde_json::json;

use super::super::AffineGatedFullAttentionConfig;
use crate::Result;

#[test]
fn derives_gated_attention_from_parsed_structure() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 128,
            "moe_intermediate_size": 64,
            "attn_output_gate": true,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 64,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 1],
                "full_attention": {
                    "partial_rotary_factor": 0.1875,
                    "rope_theta": 1_000_000.0
                }
            }
        }
    }))?;

    let config = AffineGatedFullAttentionConfig::from_decoder(&decoder, 1, 64, 4, 1.0)?;

    assert_eq!(config.query_heads, 2);
    assert_eq!(config.key_value_heads, 1);
    assert_eq!(config.head_dim, 32);
    assert_eq!(config.rotary_dim, 6);
    assert_eq!(config.rope_sections, [1, 1, 1]);
    assert!(config.rope_interleaved);
    assert!((config.rope_theta - 1_000_000.0).abs() < f32::EPSILON);
    Ok(())
}
