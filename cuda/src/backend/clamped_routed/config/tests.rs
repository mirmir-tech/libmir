use models::semantic::{
    ActivationSpec, AttentionOutputSpec, AttentionSpec, BlockLayoutSpec, DecoderLayerSpec,
    DecoderSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, NormalizationKind,
    NormalizationSpec, PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec,
    RotarySpec, RoutedExpertsSpec, RouterNormalization, SemanticModelSpec,
};

use super::ClampedRoutedConfig;

#[test]
fn admits_dynamic_geometry_instead_of_one_checkpoint_size() -> crate::Result<()> {
    let spec = spec(64, 96, 4, 2, 16);
    let config = ClampedRoutedConfig::from_admitted(&plan(&spec)?, &spec)?;

    assert_eq!(config.hidden, 64);
    assert_eq!(config.intermediate, 96);
    assert_eq!(config.top_k, 2);
    assert!(!config.rope_truncate);
    Ok(())
}

#[test]
fn semantic_config_does_not_apply_cuda_kernel_limits() -> crate::Result<()> {
    let spec = spec(64, 96, 4, 2, 258);
    let config = ClampedRoutedConfig::from_admitted(&plan(&spec)?, &spec)?;
    assert_eq!(config.head_dim, 258);
    Ok(())
}

fn plan(spec: &SemanticModelSpec) -> crate::Result<architecture::cuda::CudaDecoderPlan> {
    Ok(architecture::cuda::CudaDecoderPlan::lower(spec, None)?)
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
        weight_offset: 0.0,
    };
    SemanticModelSpec {
        schema_version: 1,
        decoder: DecoderSpec {
            hidden_size: hidden,
            vocab_size: 128,
            tie_word_embeddings: false,
            final_norm: norm,
            embedding_multiplier: 1.0,
            residual_multiplier: 1.0,
            logit_divisor: 1.0,
            layers: vec![DecoderLayerSpec {
                index: 0,
                input_norm: norm,
                post_attention_norm: norm,
                norm_placement: models::semantic::NormPlacement::Pre,
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
                    window: std::num::NonZeroUsize::new(32)
                        .map(models::semantic::AttentionWindow::Sliding),
                    position: PositionEncodingSpec::Rotary(RotarySpec {
                        theta: 150_000.0,
                        partial_factor: 1.0,
                        layout: RotaryLayoutSpec::Standard,
                        algorithm: models::semantic::RotaryAlgorithm::Default,
                        scaling: Some(RopeScalingSpec::Yarn {
                            factor: 32.0,
                            beta_fast: 32.0,
                            beta_slow: 1.0,
                            original_context_len: 4096,
                            attention_factor: 1.0,
                            truncate: false,
                        }),
                    }),
                    logit_softcap: None,
                    query_temperature: None,
                    key_value_source: None,
                }),
                feed_forward: FeedForwardSpec::Routed {
                    routed: RoutedExpertsSpec {
                        expert_count: 8,
                        top_k: 2,
                        intermediate_size: intermediate,
                        activation: ActivationSpec::ClampedSwiGlu { alpha: 1.702, limit: 7.0 },
                        router_normalization: RouterNormalization::SoftmaxTopK,
                        group_limit: None,
                        output_scale: 1.0,
                    },
                    shared: None,
                },
            }],
            block_layout: BlockLayoutSpec::Unified,
            prediction_layers: Vec::new(),
            per_layer_input: None,
        },
    }
}
