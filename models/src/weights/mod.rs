mod catalog;
mod hybrid_linear;
mod schema;

pub use catalog::{TensorCatalog, TensorInfo};
pub use schema::{DecoderTensorSchema, TensorReadiness, TensorRequirement};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::layout::{
        AttentionLayerType, AttentionOutput, DecoderConfig, LinearAttentionConfig,
        RotaryEmbeddingLayout,
    };

    #[test]
    fn reports_missing_decoder_tensors() {
        let tensors = catalog(["model.embed_tokens.weight"]);
        let schema = DecoderTensorSchema::discover(&decoder_config(), &tensors);
        let readiness = schema.readiness(&tensors);

        assert!(!readiness.is_ready());
        assert_eq!(readiness.present, 1);
        assert!(readiness.missing.iter().any(|item| item.contains("attention q")));
    }

    #[test]
    fn routed_moe_schema_requires_router_and_experts() {
        let mut config = decoder_config();
        config.num_experts = Some(128);
        config.top_k_experts = Some(8);
        let tensors = catalog(["model.embed_tokens.weight"]);
        let schema = DecoderTensorSchema::discover(&config, &tensors);
        let readiness = schema.readiness(&tensors);

        assert!(!readiness.is_ready());
        assert!(readiness.missing.iter().any(|item| item.contains("router")));
        assert!(readiness.missing.iter().any(|item| item.contains("expert gate")));
    }

    #[test]
    fn accepts_nested_language_model_and_split_expert_layout() {
        let mut config = decoder_config();
        config.attention_k_eq_v = true;
        config.num_experts = Some(128);
        config.top_k_experts = Some(8);
        let tensors = catalog([
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.router.proj.weight",
            "model.language_model.layers.0.experts.0.gate_proj.weight",
            "model.language_model.layers.0.experts.0.up_proj.weight",
            "model.language_model.layers.0.experts.0.down_proj.weight",
        ]);

        let readiness = DecoderTensorSchema::discover(&config, &tensors).readiness(&tensors);

        assert!(readiness.is_ready(), "{:?}", readiness.missing);
    }

    #[test]
    fn hybrid_linear_schema_requires_its_exact_execution_tensors() {
        let mut config = decoder_config();
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
        let tensors = hybrid_linear_catalog();
        let readiness = DecoderTensorSchema::discover(&config, &tensors).readiness(&tensors);

        assert!(readiness.is_ready());
    }

    fn catalog<const N: usize>(names: [&str; N]) -> TensorCatalog {
        TensorCatalog {
            tensors: names
                .into_iter()
                .map(|name| TensorInfo {
                    name: name.to_owned(),
                    file: PathBuf::new(),
                    dtype: "F16".into(),
                    shape: vec![],
                    data_start: 0,
                    data_offsets: [0, 0],
                })
                .collect(),
        }
    }

    fn hybrid_linear_catalog() -> TensorCatalog {
        catalog([
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.post_attention_layernorm.weight",
            "language_model.model.layers.0.linear_attn.A_log",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.0.linear_attn.dt_bias",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_z.weight",
            "language_model.model.layers.0.linear_attn.in_proj_b.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
            "language_model.model.layers.0.linear_attn.norm.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            "language_model.model.layers.0.mlp.gate.weight",
            "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
            "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
        ])
    }

    fn decoder_config() -> DecoderConfig {
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
            tie_word_embeddings: true,
            attention_k_eq_v: false,
            attention_scale: None,
            attention_output: AttentionOutput::Direct,
            sliding_window: None,
            linear_attention: None,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            hidden_activation: None,
            final_logit_softcapping: None,
        }
    }
}
