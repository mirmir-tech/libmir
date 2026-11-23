//! microsoft/Phi-3.5-MoE-instruct `config.json`: biased `LayerNorm`,
//! sparsemixer routing and `LongRoPE` with `long_mscale`.

use serde_json::json;

use super::{DecoderConfig, RopeScaling};
use crate::{
    error::{ModelsError, Result},
    layout::NormKind,
    semantic::{FeedForwardSpec, NormalizationKind, RouterNormalization, SemanticModelSpec},
    weights::TensorCatalog,
};

#[test]
fn phimoe_routes_by_sparsemixer_behind_layer_norms() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "phimoe", "hidden_size": 64, "intermediate_size": 96,
        "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 32, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "num_local_experts": 4, "num_experts_per_tok": 2, "router_jitter_noise": 0.01,
        "max_position_embeddings": 131_072, "original_max_position_embeddings": 4096,
        "rope_scaling": {
            "type": "longrope", "long_factor": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "short_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "long_mscale": 1.243, "short_mscale": 1.243,
        },
    }))?;
    assert_eq!(decoder.norm_kind, NormKind::LayerNorm);
    assert_eq!(decoder.expert_router, crate::layout::ExpertRouter::SparseMixer { jitter: 0.01 });
    let Some(RopeScaling::LongRope { attention_factor, .. }) = decoder.rope_scaling else {
        return Err(ModelsError::InvalidConfig("expected LongRoPE".into()));
    };
    assert!((attention_factor - 1.243).abs() < 1.0e-12);

    let spec = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    assert_eq!(spec.decoder.final_norm.kind, NormalizationKind::Layer);
    let FeedForwardSpec::Routed { routed, shared: None } = &spec.decoder.layers[0].feed_forward
    else {
        return Err(ModelsError::InvalidConfig("expected routed experts".into()));
    };
    assert_eq!(
        routed.router_normalization,
        RouterNormalization::SparseMixer { jitter_eps: 0.01 }
    );
    Ok(())
}
