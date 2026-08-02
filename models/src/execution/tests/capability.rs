use serde_json::json;

use crate::{
    execution::{ArchitectureCapability, ArchitectureRequirements, TaskExecutionPlan},
    layout::DecoderConfig,
    semantic::{
        ActivationSpec, AttentionOutputSpec, FeedForwardSpec, MixerSpec, NormalizationKind,
        PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec,
        RouterNormalization, SemanticModelSpec,
    },
    weights::{TensorCatalog, TensorInfo},
};

#[test]
fn derives_deduplicated_requirements_from_semantics_and_task() -> crate::Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "hidden_act": "silu"
    }))?;
    let task = TaskExecutionPlan::Generation { decoder: decoder.clone() };
    let catalog = TensorCatalog {
        tensors: [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ]
        .into_iter()
        .map(tensor)
        .collect(),
    };
    let mut semantic = SemanticModelSpec::discover(&decoder, &catalog)?;
    let layer = &mut semantic.decoder.layers[0];
    if let MixerSpec::SoftmaxAttention(attention) = &mut layer.mixer {
        attention.sinks = true;
        attention.window = Some(128);
        attention.qk_normalization = QkNormalization::QueryKeyRms;
        attention.output = AttentionOutputSpec::Gated;
        if let PositionEncodingSpec::Rotary(rotary) = &mut attention.position {
            rotary.layout = RotaryLayoutSpec::MultiSection(vec![2, 1, 1]);
            rotary.scaling = Some(RopeScalingSpec::Yarn {
                factor: 4.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_context_len: 4096,
                attention_factor: 1.0,
            });
        }
    }
    layer.feed_forward = FeedForwardSpec::DenseAndRouted {
        dense_intermediate_size: 16,
        dense_activation: ActivationSpec::GeluTanh,
        routed: crate::semantic::RoutedExpertsSpec {
            expert_count: 8,
            top_k: 2,
            intermediate_size: 16,
            activation: ActivationSpec::SwiGlu {
                alpha: 1.0,
                clamp: Some(7.0),
                up_shift: 1.0,
            },
            router_normalization: RouterNormalization::UnitTopK,
        },
    };
    semantic.decoder.final_norm.kind = NormalizationKind::Layer;

    let requirements = ArchitectureRequirements::discover(&task, Some(&semantic));
    for capability in [
        ArchitectureCapability::GenerationTask,
        ArchitectureCapability::LayerNormalization,
        ArchitectureCapability::AttentionSinks,
        ArchitectureCapability::SlidingWindowAttention,
        ArchitectureCapability::QueryKeyRmsNormalization,
        ArchitectureCapability::GatedAttentionOutput,
        ArchitectureCapability::MultiSectionRotary,
        ArchitectureCapability::YarnRopeScaling,
        ArchitectureCapability::DenseAndRouted,
        ArchitectureCapability::RoutedExperts,
        ArchitectureCapability::ClampedSwiGlu,
        ArchitectureCapability::GeluTanh,
        ArchitectureCapability::UnitTopKRouter,
    ] {
        assert!(requirements.capabilities.contains(&capability), "missing {capability:?}");
    }
    Ok(())
}

fn tensor(name: &str) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: std::path::PathBuf::new(),
        dtype: "BF16".into(),
        shape: vec![8],
        data_start: 0,
        data_offsets: [0, 0],
    }
}
