use std::path::PathBuf;

use super::*;
use crate::{
    layout::{
        AttentionLayerType, AttentionOutput, DecoderConfig, LinearAttentionConfig,
        RotaryEmbeddingLayout,
    },
    weights::TensorInfo,
};

#[test]
fn discovers_dense_swiglu_without_a_model_family() -> Result<()> {
    let plan = ExecutionPlan::discover(&decoder(), &catalog_with_output_head(DENSE_SWIGLU_LAYOUT))?;

    assert_eq!(plan.decoder, DecoderArchetype::DenseSwiGlu);
    assert_eq!(plan.feed_forward, FeedForwardFeature::DenseSwiGlu);
    Ok(())
}

#[test]
fn discovers_dense_swiglu_with_tied_embeddings() -> Result<()> {
    let mut config = decoder();
    config.tie_word_embeddings = true;

    let plan = ExecutionPlan::discover(&config, &catalog(DENSE_SWIGLU_LAYOUT))?;

    assert_eq!(plan.decoder, DecoderArchetype::DenseSwiGlu);
    Ok(())
}

#[test]
fn discovers_normalized_grouped_query_attention() -> Result<()> {
    let mut names = DENSE_SWIGLU_LAYOUT.to_vec();
    names.extend(DENSE_QK_NORM_LAYOUT);
    names.push("lm_head.weight");

    let plan = ExecutionPlan::discover(&decoder(), &catalog(&names))?;

    assert_eq!(plan.attention, AttentionFeature::RmsNormalizedGroupedQuery);
    Ok(())
}

#[test]
fn rejects_hybrid_layout_when_features_do_not_match() {
    let error = ExecutionPlan::discover(&decoder(), &catalog(HYBRID_MOE_LAYOUT));

    assert!(error.is_err());
}

#[test]
fn discovers_routed_moe_from_decoder_features_and_tensor_layout() -> Result<()> {
    let mut config = decoder();
    config.attention_k_eq_v = true;
    config.num_experts = Some(8);
    config.top_k_experts = Some(2);
    config.hidden_activation = Some("gelu_pytorch_tanh".into());

    let plan = ExecutionPlan::discover(&config, &catalog(HYBRID_MOE_LAYOUT))?;

    assert_eq!(plan.decoder, DecoderArchetype::HybridMoe);
    assert_eq!(plan.attention, AttentionFeature::RmsNormalizedSharedKv);
    assert_eq!(plan.feed_forward, FeedForwardFeature::DenseGeluAndRoutedMoe);
    Ok(())
}

#[test]
fn discovers_hybrid_linear_moe_from_features_and_tensor_layout() -> Result<()> {
    let mut config = decoder();
    config.num_experts = Some(8);
    config.top_k_experts = Some(2);
    config.shared_expert_intermediate_size = Some(8);
    config.attention_output = AttentionOutput::Gated;
    config.linear_attention = Some(LinearAttentionConfig {
        convolution_kernel_size: 4,
        key_heads: 2,
        value_heads: 4,
        key_head_dim: 2,
        value_head_dim: 2,
    });
    config.layer_types = vec![AttentionLayerType::Linear, AttentionLayerType::Full];

    let plan = ExecutionPlan::discover(&config, &catalog(HYBRID_LINEAR_MOE_LAYOUT))?;

    assert_eq!(plan.decoder, DecoderArchetype::HybridLinearMoe);
    assert_eq!(plan.attention, AttentionFeature::GatedDeltaAndRmsNormalizedGroupedQuery);
    assert_eq!(plan.feed_forward, FeedForwardFeature::SharedExpertRoutedSwiGlu);
    assert!(plan.is_native_implemented());
    Ok(())
}

fn catalog(names: &[&str]) -> TensorCatalog {
    TensorCatalog {
        tensors: names
            .iter()
            .map(|name| TensorInfo {
                name: (*name).into(),
                file: PathBuf::new(),
                dtype: "U32".into(),
                shape: Vec::new(),
                data_start: 0,
                data_offsets: [0, 0],
            })
            .collect(),
    }
}

fn catalog_with_output_head(names: &[&str]) -> TensorCatalog {
    let mut names = names.to_vec();
    names.push("lm_head.weight");
    catalog(&names)
}

fn decoder() -> DecoderConfig {
    DecoderConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 2,
        global_head_dim: None,
        num_global_key_value_heads: None,
        vocab_size: 16,
        max_position_embeddings: 8,
        rms_norm_eps: 1e-5,
        rope_theta: None,
        rope_scaling: None,
        partial_rotary_factor: None,
        rope_layout: RotaryEmbeddingLayout::Standard,
        full_attention_rope_theta: None,
        sliding_attention_rope_theta: None,
        full_attention_rope_type: None,
        sliding_attention_rope_type: None,
        full_attention_partial_rotary_factor: None,
        sliding_attention_partial_rotary_factor: None,
        layer_types: vec![AttentionLayerType::Full],
        tie_word_embeddings: false,
        attention_k_eq_v: false,
        attention_scale: None,
        attention_output: AttentionOutput::Direct,
        sliding_window: None,
        linear_attention: None,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        hidden_activation: Some("silu".into()),
        final_logit_softcapping: None,
    }
}
