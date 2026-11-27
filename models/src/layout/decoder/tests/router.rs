//! A softmax router renormalizes its selected experts' probabilities as
//! `norm_topk_prob` declares, else as transformers' configuration of the
//! schema defaults: Qwen1.5-MoE (`qwen2_moe`) declares false and keeps
//! them; Mixtral declares nothing and renormalizes.

use serde_json::{Value, json};

use super::DecoderConfig;
use crate::{error::Result, semantic::RouterNormalization};

fn router(model_type: &str, norm_topk_prob: Option<bool>) -> Result<RouterNormalization> {
    let mut config = json!({
        "model_type": model_type, "hidden_size": 64, "intermediate_size": 96,
        "moe_intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 32, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "num_experts": 4, "num_experts_per_tok": 2,
    });
    if let (Some(declared), Value::Object(config)) = (norm_topk_prob, &mut config) {
        config.insert("norm_topk_prob".into(), json!(declared));
    }
    Ok(DecoderConfig::from_value(&config)?.expert_router)
}

#[test]
fn a_softmax_router_renormalizes_as_declared_else_as_its_schema() -> Result<()> {
    use RouterNormalization::{SoftmaxProbabilityTopK as Kept, SoftmaxTopK as Renormalized};
    assert_eq!(router("qwen2_moe", Some(false))?, Kept);
    assert_eq!(router("qwen3_moe", Some(true))?, Renormalized);
    assert_eq!(router("mixtral", None)?, Renormalized);
    for model_type in ["qwen2_moe", "qwen3_moe", "olmoe"] {
        assert_eq!(router(model_type, None)?, Kept, "{model_type}");
    }
    assert_eq!(router("mixtral", Some(false))?, Kept);
    Ok(())
}
