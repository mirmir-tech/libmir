//! Latent-attention decoders: the attention and expert fields of `DeepSeek` V2
//! Lite, `Moonlight` and Mistral Small 4 `config.json` files (2026-11-20).

use serde_json::{Value, json};

use super::*;
use crate::{
    layout::{AttentionLayerType, ExpertRouting, RopeScaling, RotaryPairing},
    semantic::ExpertGroupLimit,
};

fn deepseek_v2_lite() -> Value {
    json!({
        "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
        "hidden_size": 2048, "intermediate_size": 10944, "moe_intermediate_size": 1408,
        "num_hidden_layers": 27, "num_attention_heads": 16, "num_key_value_heads": 16,
        "vocab_size": 102_400, "q_lora_rank": null, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "n_routed_experts": 64, "num_experts_per_tok": 6, "n_shared_experts": 2,
        "first_k_dense_replace": 1, "moe_layer_freq": 1, "n_group": 1, "topk_group": 1,
        "topk_method": "greedy", "scoring_func": "softmax", "norm_topk_prob": false,
        "routed_scaling_factor": 1.0, "rope_theta": 10000,
        "rope_scaling": {
            "beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 0.707,
            "mscale_all_dim": 0.707, "original_max_position_embeddings": 4096, "type": "yarn"
        },
        "max_position_embeddings": 163_840, "hidden_act": "silu", "rms_norm_eps": 1e-6
    })
}

fn moonlight() -> Value {
    json!({
        "architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3",
        "hidden_size": 2048, "intermediate_size": 11264, "moe_intermediate_size": 1408,
        "num_hidden_layers": 27, "num_attention_heads": 16, "num_key_value_heads": 16,
        "vocab_size": 163_840, "q_lora_rank": null, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "n_routed_experts": 64, "num_experts_per_tok": 6, "n_shared_experts": 2,
        "first_k_dense_replace": 1, "moe_layer_freq": 1, "n_group": 1, "topk_group": 1,
        "topk_method": "noaux_tc", "scoring_func": "sigmoid", "norm_topk_prob": true,
        "routed_scaling_factor": 2.446, "rope_theta": 50000.0, "rope_scaling": null,
        "max_position_embeddings": 8192, "hidden_act": "silu", "rms_norm_eps": 1e-5
    })
}

fn mistral_small_4() -> Value {
    json!({
        "architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
        "text_config": {
            "model_type": "mistral4", "hidden_size": 4096, "intermediate_size": 12288,
            "moe_intermediate_size": 2048, "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 32, "head_dim": 128, "vocab_size": 131_072,
            "q_lora_rank": 1024, "kv_lora_rank": 256, "qk_head_dim": 128,
            "qk_nope_head_dim": 64, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "n_routed_experts": 128, "num_experts_per_tok": 4, "n_shared_experts": 1,
            "first_k_dense_replace": 0, "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "routed_scaling_factor": 1.0, "rope_interleave": true, "sliding_window": null,
            "rope_parameters": {
                "beta_fast": 32.0, "beta_slow": 1.0, "factor": 128.0,
                "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 8192, "rope_theta": 10000.0,
                "rope_type": "yarn", "type": "yarn"
            },
            "max_position_embeddings": 1_048_576, "hidden_act": "silu", "rms_norm_eps": 1e-6
        }
    })
}

fn invalid(message: &str) -> crate::ModelsError {
    crate::ModelsError::InvalidConfig(message.into())
}

fn yarn_mscale(factor: f64, all_dim: f64) -> f64 {
    0.1_f64.mul_add(all_dim * factor.ln(), 1.0)
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-12
}

#[test]
fn reads_deepseek_v2_lite_latent_attention() -> Result<()> {
    let decoder = DecoderConfig::from_value(&deepseek_v2_lite())?;
    let latent = decoder.latent.as_ref().ok_or_else(|| invalid("missing latent attention"))?;
    let attention = &latent.attention;

    assert_eq!(attention.query_rank, None);
    assert_eq!((attention.key_value_rank, attention.cached_width()), (512, 576));
    assert_eq!((decoder.head_dim, attention.rotary_pairing), (192, RotaryPairing::Interleaved));
    assert!(close(
        attention.scale,
        192_f64.sqrt().recip() * yarn_mscale(40.0, 0.707).powi(2)
    ));
    assert!(decoder.layer_types.iter().all(|layer| *layer == AttentionLayerType::Latent));
    assert!(
        matches!(decoder.rope_scaling, Some(RopeScaling::Yarn { factor, .. }) if close(factor, 40.0))
    );
    let experts = latent.experts.as_ref().ok_or_else(|| invalid("missing experts"))?;
    assert_eq!(experts.first_dense_layers, 1);
    assert_eq!(experts.shared_experts.map(std::num::NonZeroUsize::get), Some(2));
    assert_eq!(experts.routing, ExpertRouting::Softmax { renormalize: false });
    Ok(())
}

#[test]
fn reads_moonlight_sigmoid_routing() -> Result<()> {
    let decoder = DecoderConfig::from_value(&moonlight())?;
    let latent = decoder.latent.as_ref().ok_or_else(|| invalid("missing latent attention"))?;
    let experts = latent.experts.as_ref().ok_or_else(|| invalid("missing experts"))?;

    assert!(close(latent.attention.scale, 192_f64.sqrt().recip()));
    assert_eq!(experts.routing, ExpertRouting::SigmoidWithCorrectionBias);
    assert!(close(experts.output_scale, 2.446));
    Ok(())
}

#[test]
fn reads_mistral_small_4_from_its_text_config_and_rope_parameters() -> Result<()> {
    let decoder = DecoderConfig::from_value(&mistral_small_4())?;
    let latent = decoder.latent.as_ref().ok_or_else(|| invalid("missing latent attention"))?;

    assert_eq!(latent.attention.query_rank.map(std::num::NonZeroUsize::get), Some(1024));
    assert_eq!(latent.attention.cached_width(), 320);
    assert!(close(
        latent.attention.scale,
        128_f64.sqrt().recip() * yarn_mscale(128.0, 1.0).powi(2)
    ));
    assert!(
        matches!(decoder.rope_scaling, Some(RopeScaling::Yarn { factor, .. }) if close(factor, 128.0))
    );
    let experts = latent.experts.as_ref().ok_or_else(|| invalid("missing experts"))?;
    assert_eq!(
        (experts.first_dense_layers, experts.routing),
        (0, ExpertRouting::Softmax { renormalize: true })
    );
    Ok(())
}

#[test]
fn reads_group_limited_expert_selection() -> Result<()> {
    let group_limit = |config: &Value| {
        let decoder = DecoderConfig::from_value(config)?;
        let latent = decoder.latent.ok_or_else(|| invalid("missing latent decoder"))?;
        let experts = latent.experts.ok_or_else(|| invalid("missing experts"))?;
        Ok::<_, crate::error::ModelsError>(experts.group_limit)
    };
    // `noaux_tc` ranks groups whenever there are several.
    let mut sigmoid = moonlight();
    sigmoid["n_group"] = json!(8);
    sigmoid["topk_group"] = json!(4);
    assert_eq!(group_limit(&sigmoid)?, Some(ExpertGroupLimit { groups: 8, selected: 4 }));
    // Softmax routers rank them only under `group_limited_greedy`.
    let mut softmax = deepseek_v2_lite();
    softmax["n_group"] = json!(8);
    softmax["topk_group"] = json!(3);
    assert_eq!(group_limit(&softmax)?, None);
    softmax["topk_method"] = json!("group_limited_greedy");
    assert_eq!(group_limit(&softmax)?, Some(ExpertGroupLimit { groups: 8, selected: 3 }));
    assert_eq!(group_limit(&deepseek_v2_lite())?, None);
    softmax["topk_method"] = json!("noaux_tc");
    assert!(group_limit(&softmax).is_err());
    Ok(())
}
