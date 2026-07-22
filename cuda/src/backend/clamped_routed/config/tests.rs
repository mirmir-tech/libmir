use models::semantic::{
    ActivationSpec, AttentionOutputSpec, AttentionSpec, DecoderLayerSpec, DecoderSpec,
    FeedForwardSpec, KeyValueRelation, MixerSpec, NormalizationKind, NormalizationSpec,
    PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec, RotarySpec,
    RoutedExpertsSpec, RouterNormalization, SemanticModelSpec,
};

use super::ClampedRoutedConfig;

#[test]
fn admits_dynamic_geometry_instead_of_one_checkpoint_size() -> crate::Result<()> {
    let spec = spec(64, 96, 4, 2, 16);
    let config = ClampedRoutedConfig::from_semantic(&spec)?;

    assert_eq!(config.hidden, 64);
    assert_eq!(config.intermediate, 96);
    assert_eq!(config.top_k, 2);
    Ok(())
}

#[test]
fn semantic_config_does_not_apply_cuda_kernel_limits() -> crate::Result<()> {
    let config = ClampedRoutedConfig::from_semantic(&spec(64, 96, 4, 2, 258))?;
    assert_eq!(config.head_dim, 258);
    Ok(())
}

fn spec(
    hidden: usize,
    intermediate: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> SemanticModelSpec {
    let norm = NormalizationSpec {
        kind: NormalizationKind::Rms,
        epsilon: 1.0e-5,
    };
    SemanticModelSpec {
        schema_version: 1,
        decoder: DecoderSpec {
            hidden_size: hidden,
            vocab_size: 128,
            tie_word_embeddings: false,
            final_norm: norm,
            layers: vec![DecoderLayerSpec {
                index: 0,
                input_norm: norm,
                post_attention_norm: norm,
                mixer: MixerSpec::SoftmaxAttention(AttentionSpec {
                    query_heads,
                    key_value_heads,
                    head_dim,
                    key_value_relation: KeyValueRelation::Separate,
                    qk_normalization: QkNormalization::None,
                    projection_bias: true,
                    output: AttentionOutputSpec::Direct,
                    sinks: true,
                    scale: 0.25,
                    window: Some(32),
                    position: PositionEncodingSpec::Rotary(RotarySpec {
                        theta: 150_000.0,
                        partial_factor: 1.0,
                        layout: RotaryLayoutSpec::Standard,
                        algorithm: None,
                        scaling: Some(RopeScalingSpec::Yarn {
                            factor: 32.0,
                            beta_fast: 32.0,
                            beta_slow: 1.0,
                            original_context_len: 4096,
                            attention_factor: 1.0,
                        }),
                    }),
                }),
                feed_forward: FeedForwardSpec::Routed {
                    routed: RoutedExpertsSpec {
                        expert_count: 8,
                        top_k: 2,
                        intermediate_size: intermediate,
                        activation: ActivationSpec::SwiGlu {
                            alpha: 1.702,
                            clamp: Some(7.0),
                            up_shift: 1.0,
                        },
                        router_normalization: RouterNormalization::UnitTopK,
                    },
                    shared: None,
                },
            }],
        },
    }
}
