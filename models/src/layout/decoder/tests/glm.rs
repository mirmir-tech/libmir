//! zai-org/GLM-4.5-Air `config.json`: the `DeepSeek` expert layout beside
//! plain grouped-query attention, scored by sigmoid without declaring it.

use std::num::NonZeroUsize;

use serde_json::json;

use super::{DecoderConfig, ExpertRouting};
use crate::{
    error::{ModelsError, Result},
    semantic::{FeedForwardSpec, RouterNormalization, SemanticModelSpec},
    weights::TensorCatalog,
};

fn glm_4_5_air() -> serde_json::Value {
    json!({
        "model_type": "glm4_moe", "hidden_size": 64, "intermediate_size": 160,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "partial_rotary_factor": 0.5, "vocab_size": 32,
        "attention_bias": true, "hidden_act": "silu",
        "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "moe_intermediate_size": 32, "first_k_dense_replace": 1, "norm_topk_prob": true,
        "routed_scaling_factor": 1.0, "n_group": 1, "topk_group": 1
    })
}

#[test]
fn glm_4_5_routes_by_sigmoid_after_its_leading_dense_layer() -> Result<()> {
    let decoder = DecoderConfig::from_value(&glm_4_5_air())?;
    let experts = decoder
        .routed_experts
        .as_ref()
        .ok_or_else(|| ModelsError::InvalidConfig("missing experts".into()))?;
    assert_eq!(experts.routing, ExpertRouting::SigmoidWithCorrectionBias);
    assert_eq!(experts.first_dense_layers, 1);
    assert_eq!(experts.shared_experts, NonZeroUsize::new(1));
    assert_eq!(experts.group_limit, None);
    assert!(decoder.latent.is_none());

    let spec = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    assert!(matches!(spec.decoder.layers[0].feed_forward, FeedForwardSpec::Dense { .. }));
    let FeedForwardSpec::Routed { routed, shared: Some(shared) } =
        &spec.decoder.layers[1].feed_forward
    else {
        return Err(ModelsError::InvalidConfig("layer 1 is not shared-routed".into()));
    };
    assert_eq!(routed.router_normalization, RouterNormalization::SigmoidBiasedTopK);
    assert_eq!((routed.intermediate_size, shared.intermediate_size), (32, 32));
    assert!(!shared.gated_output);
    Ok(())
}
