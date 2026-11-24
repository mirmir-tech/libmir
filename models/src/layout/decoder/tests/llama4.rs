//! meta-llama/Llama-4-Scout-17B-16E-Instruct `text_config`: every fourth
//! layer attends without rotation at a query temperature, the others in
//! 8192-token chunks with unit Q/K norms; one sigmoid-scaled expert and an
//! ungated shared expert per layer.

use serde_json::json;

use super::DecoderConfig;
use crate::{
    error::{ModelsError, Result},
    semantic::{
        AttentionWindow, FeedForwardSpec, MixerSpec, PositionEncodingSpec, QkNormalization,
        RouterNormalization, SemanticModelSpec,
    },
    weights::TensorCatalog,
};

#[test]
fn llama4_alternates_chunked_rotary_and_tempered_global_layers() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "llama4_text", "hidden_size": 64, "intermediate_size": 32,
        "intermediate_size_mlp": 96, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 32, "hidden_act": "silu",
        "num_local_experts": 4, "num_experts_per_tok": 1, "interleave_moe_layer_step": 1,
        "attention_chunk_size": 8192, "attn_temperature_tuning": true, "floor_scale": 8192,
        "attn_scale": 0.1, "use_qk_norm": true, "rope_theta": 500_000.0,
    }))?;
    assert_eq!(decoder.expert_router, RouterNormalization::SigmoidInputScaledTopK);
    assert!(decoder.query_key_rows_interleaved);
    let spec = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    let attention = |index: usize| match &spec.decoder.layers[index].mixer {
        MixerSpec::SoftmaxAttention(attention) => Ok(attention.clone()),
        _ => Err(ModelsError::InvalidConfig("expected softmax attention".into())),
    };
    let rotary = attention(0)?;
    assert_eq!(
        rotary.window,
        std::num::NonZeroUsize::new(8192).map(|chunk| AttentionWindow::Chunked { chunk })
    );
    assert_eq!(rotary.qk_normalization, QkNormalization::QueryKeyUnitRms);
    assert!(rotary.query_temperature.is_none());
    let global = attention(3)?;
    assert_eq!(global.position, PositionEncodingSpec::None);
    assert_eq!(global.window, None);
    assert_eq!(global.qk_normalization, QkNormalization::None);
    let temperature = global
        .query_temperature
        .ok_or(ModelsError::InvalidConfig("temperature".into()))?;
    assert_eq!((temperature.first_scaled_position(), temperature.factor(8190)), (8191, 1.0));
    let FeedForwardSpec::Routed { routed, shared: Some(shared) } =
        &spec.decoder.layers[0].feed_forward
    else {
        return Err(ModelsError::InvalidConfig("expected shared-routed experts".into()));
    };
    assert_eq!(routed.router_normalization, RouterNormalization::SigmoidInputScaledTopK);
    assert_eq!((shared.intermediate_size, shared.gated_output), (32, false));
    Ok(())
}
