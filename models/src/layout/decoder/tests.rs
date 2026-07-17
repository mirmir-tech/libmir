use serde_json::json;

use super::*;
use crate::error::Result;

#[test]
fn reads_common_decoder_fields() -> Result<()> {
    let value = json!({
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 0.000_001,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true
    });
    let config = DecoderConfig::from_value(&value)?;

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.num_key_value_heads, 8);
    assert!(config.tie_word_embeddings);
    assert!(!config.attention_k_eq_v);
    assert_eq!(config.layer_types, vec![AttentionLayerType::Full; 32]);
    Ok(())
}

#[test]
fn reads_nested_text_config() -> Result<()> {
    let value = json!({
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "hidden_size": 2816,
            "intermediate_size": 2112,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 2,
            "vocab_size": 262_144,
            "attention_k_eq_v": true,
            "sliding_window": 1024,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0,
                    "rope_type": "default"
                }
            }
        }
    });
    let config = DecoderConfig::from_value(&value)?;

    assert_eq!(config.hidden_size, 2816);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.global_head_dim, Some(512));
    assert_eq!(config.num_global_key_value_heads, Some(2));
    assert_eq!(config.sliding_attention_rope_theta, Some(10_000.0));
    assert_eq!(config.full_attention_rope_theta, Some(1_000_000.0));
    assert_eq!(config.full_attention_rope_type.as_deref(), Some("proportional"));
    assert_eq!(config.sliding_attention_rope_type.as_deref(), Some("default"));
    assert_eq!(config.full_attention_partial_rotary_factor, Some(0.25));
    assert_eq!(config.rope_theta_for_layer(0), Some(10_000.0));
    assert_eq!(config.rope_theta_for_layer(1), Some(1_000_000.0));
    assert_eq!(config.layer_types, vec![AttentionLayerType::Sliding, AttentionLayerType::Full]);
    assert!(config.tie_word_embeddings);
    assert!(config.attention_k_eq_v);
    assert_eq!(config.attention_scale, Some(1.0));
    assert_eq!(config.sliding_window, Some(1024));
    assert_eq!(config.num_experts, Some(128));
    assert_eq!(config.top_k_experts, Some(8));
    assert_eq!(config.moe_intermediate_size, Some(704));
    assert_eq!(config.hidden_activation.as_deref(), Some("gelu_pytorch_tanh"));
    assert_eq!(config.final_logit_softcapping, Some(30.0));
    Ok(())
}

#[test]
fn reads_chatglm_kv_channels_as_head_dim() -> Result<()> {
    let value = json!({
        "model_type": "chatglm",
        "hidden_size": 4,
        "ffn_hidden_size": 6,
        "num_layers": 2,
        "num_attention_heads": 4,
        "multi_query_group_num": 2,
        "kv_channels": 2,
        "vocab_size": 65024,
        "seq_length": 131_072,
        "layernorm_epsilon": 0.00001
    });
    let config = DecoderConfig::from_value(&value)?;

    assert_eq!(config.head_dim, 2);
    assert_eq!(config.intermediate_size, 6);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.max_position_embeddings, 131_072);
    Ok(())
}

#[test]
fn reads_piecewise_rope_scaling() -> Result<()> {
    let value = json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 32,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
    });
    let config = DecoderConfig::from_value(&value)?;

    assert_eq!(
        config.rope_scaling,
        Some(RopeScaling::PiecewiseFrequency {
            factor: 32.0,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
            original_context_len: 8192,
        })
    );
    Ok(())
}

#[test]
fn accepts_null_optional_decoder_fields() -> Result<()> {
    let value = json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "vocab_size": 32,
        "sliding_window": null,
        "rope_scaling": null
    });
    let config = DecoderConfig::from_value(&value)?;

    assert_eq!(config.sliding_window, None);
    assert_eq!(config.rope_scaling, None);
    Ok(())
}

#[test]
fn reads_hybrid_linear_moe_features() -> Result<()> {
    let value = json!({
        "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "attn_output_gate": true,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "rope_parameters": {
                "rope_theta": 1_000_000.0,
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 1]
            }
        }
    });
    let config = DecoderConfig::from_value(&value)?;

    assert_eq!(config.intermediate_size, 16);
    assert_eq!(config.layer_type(0), AttentionLayerType::Linear);
    assert_eq!(config.layer_type(3), AttentionLayerType::Full);
    assert_eq!(config.shared_expert_intermediate_size, Some(16));
    assert_eq!(config.attention_output, AttentionOutput::Gated);
    assert_eq!(
        config.rope_layout,
        RotaryEmbeddingLayout::InterleavedMultiSection(vec![1, 1, 1])
    );
    assert_eq!(config.partial_rotary_factor, None);
    assert_eq!(
        config.linear_attention,
        Some(LinearAttentionConfig {
            convolution_kernel_size: 4,
            key_heads: 4,
            value_heads: 8,
            key_head_dim: 4,
            value_head_dim: 4,
        })
    );
    Ok(())
}
