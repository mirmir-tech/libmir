use std::path::PathBuf;

use models::{
    execution::TaskExecutionPlan,
    layout::DecoderConfig,
    semantic::{
        ActivationSpec, AttentionOutputSpec, FeedForwardSpec, LinearAttentionSpec, MixerSpec,
        RoutedExpertsSpec, RouterNormalization, SemanticModelSpec, SharedExpertSpec,
    },
    weights::{TensorCatalog, TensorInfo},
};
use serde_json::json;

pub(super) type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

pub(super) fn dense_contract() -> AnyResult<(TaskExecutionPlan, SemanticModelSpec)> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "hidden_act": "silu"
    }))?;
    let catalog = TensorCatalog {
        tensors: vec![TensorInfo {
            name: "model.layers.0.input_layernorm.weight".into(),
            file: PathBuf::new(),
            dtype: "BF16".into(),
            shape: vec![8],
            data_start: 0,
            data_offsets: [0, 0],
        }],
    };
    let semantic = SemanticModelSpec::discover(&decoder, &catalog)?;
    Ok((TaskExecutionPlan::Generation { decoder }, semantic))
}

pub(super) fn dense_and_routed(mut semantic: SemanticModelSpec) -> SemanticModelSpec {
    for layer in &mut semantic.decoder.layers {
        layer.feed_forward = FeedForwardSpec::DenseAndRouted {
            dense_intermediate_size: 16,
            dense_activation: ActivationSpec::GeluTanh,
            routed: routed(ActivationSpec::GeluTanh),
        };
    }
    semantic
}

pub(super) fn clamped_routed(mut semantic: SemanticModelSpec) -> SemanticModelSpec {
    for layer in &mut semantic.decoder.layers {
        let MixerSpec::SoftmaxAttention(attention) = &mut layer.mixer else {
            continue;
        };
        attention.sinks = true;
        layer.feed_forward = FeedForwardSpec::Routed {
            routed: routed(ActivationSpec::SwiGlu {
                alpha: 1.0,
                clamp: Some(7.0),
                up_shift: 1.0,
            }),
            shared: None,
        };
    }
    semantic
}

pub(super) fn shared_routed(mut semantic: SemanticModelSpec) -> SemanticModelSpec {
    for layer in &mut semantic.decoder.layers {
        layer.feed_forward = FeedForwardSpec::Routed {
            routed: routed(swiglu()),
            shared: Some(SharedExpertSpec {
                intermediate_size: 16,
                activation: swiglu(),
                gated_output: true,
            }),
        };
    }
    semantic.decoder.layers[0].mixer = MixerSpec::LinearAttention(LinearAttentionSpec {
        convolution_kernel_size: 4,
        key_heads: 2,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        output: AttentionOutputSpec::Direct,
    });
    semantic
}

pub(super) fn mixed_unsupported(mut semantic: SemanticModelSpec) -> SemanticModelSpec {
    semantic.decoder.layers[1].feed_forward = FeedForwardSpec::DenseAndRouted {
        dense_intermediate_size: 16,
        dense_activation: ActivationSpec::GeluTanh,
        routed: routed(ActivationSpec::GeluTanh),
    };
    semantic
}

fn routed(activation: ActivationSpec) -> RoutedExpertsSpec {
    RoutedExpertsSpec {
        expert_count: 8,
        top_k: 2,
        intermediate_size: 16,
        activation,
        router_normalization: RouterNormalization::SoftmaxTopK,
    }
}

fn swiglu() -> ActivationSpec {
    ActivationSpec::SwiGlu { alpha: 1.0, clamp: None, up_shift: 0.0 }
}
