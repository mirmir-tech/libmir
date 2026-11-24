use architecture::{ArchitectureError, Unlowerable};
use models::{
    execution::TaskExecutionPlan,
    layout::DecoderConfig,
    semantic::{
        ActivationSpec, AttentionOutputSpec, FeedForwardSpec, KeyValueRelation,
        LinearAttentionSpec, MixerSpec, NormalizationKind, SemanticModelSpec,
    },
    weights::{TensorCatalog, TensorInfo},
};
use serde_json::json;

use super::*;
use crate::engine::Error;

#[test]
fn materializes_dense_layers_without_model_identity() -> Result<()> {
    let spec = dense_spec()?;
    let lowering = plan(&dense_decoder()?, &spec)?;

    assert_eq!(
        lowering.layers(),
        &[LayerLowering {
            index: 0,
            input_norm: NormalizationLowering::Rms,
            post_attention_norm: NormalizationLowering::Rms,
            norm_placement: models::semantic::NormPlacement::Pre,
            mixer: MixerLowering::Softmax {
                normalization: models::semantic::QkNormalization::None,
                sinks: false,
                window: None,
                key_value_relation: KeyValueRelation::Separate,
            },
            feed_forward: FeedForwardLowering::Dense {
                activation: Activation::SwiGlu { clamp: None }
            },
        }]
    );
    assert_eq!(lowering.runtime(), DecoderRuntime::Dense);
    assert_eq!(
        crate::admit_architecture(
            &TaskExecutionPlan::Generation { decoder: dense_decoder()? },
            Some(&spec),
        )?,
        crate::MetalArchitecture::Generation(lowering)
    );
    Ok(())
}

#[test]
fn rejects_an_unavailable_normalization_independently() -> Result<()> {
    let mut spec = dense_spec()?;
    spec.decoder.layers[0].input_norm.kind = NormalizationKind::Layer;
    spec.decoder.layers[0].input_norm.weight_offset = 1.0;

    assert!(matches!(
        lower_layer(&spec.decoder.layers[0]),
        Err(Error::Architecture(ArchitectureError::Unlowerable {
            layer: Some(0),
            feature: Unlowerable::LayerNormWeightOffset,
        }))
    ));
    Ok(())
}

#[test]
fn admits_linear_attention_through_the_dense_mixed_runtime() -> Result<()> {
    let mut spec = dense_spec()?;
    spec.decoder.layers[0].mixer = MixerSpec::LinearAttention(LinearAttentionSpec {
        convolution_kernel_size: 4,
        key_heads: 2,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        output: AttentionOutputSpec::Direct,
    });

    assert_eq!(lower_layer(&spec.decoder.layers[0])?.mixer, MixerLowering::Linear);
    assert_eq!(plan(&dense_decoder()?, &spec)?.runtime(), DecoderRuntime::DenseMixed);
    Ok(())
}

#[test]
fn preserves_windows_for_the_dense_runtime() -> Result<()> {
    let mut spec = dense_spec()?;
    if let MixerSpec::SoftmaxAttention(attention) = &mut spec.decoder.layers[0].mixer {
        attention.window =
            std::num::NonZeroUsize::new(128).map(models::semantic::AttentionWindow::Sliding);
    }

    assert_eq!(
        lower_layer(&spec.decoder.layers[0])?.mixer,
        MixerLowering::Softmax {
            normalization: models::semantic::QkNormalization::None,
            sinks: false,
            window: std::num::NonZeroUsize::new(128)
                .map(models::semantic::AttentionWindow::Sliding),
            key_value_relation: KeyValueRelation::Separate,
        }
    );
    // The dense runtime runs sliding layers (Gemma 3) with the cache's mask.
    assert_eq!(plan(&dense_decoder()?, &spec)?.runtime(), DecoderRuntime::Dense);
    Ok(())
}

#[test]
fn rejects_an_activation_the_dense_operator_does_not_implement() -> Result<()> {
    let mut spec = dense_spec()?;
    // Tanh GELU (Gemma's GeGLU) lowers; an ungated activation lowers to the
    // relu² MLP, which only the state-space hybrid runtime executes, beside
    // its Mamba mixers.
    if let FeedForwardSpec::Dense { activation, .. } = &mut spec.decoder.layers[0].feed_forward {
        *activation = ActivationSpec::GeluTanh;
    }
    assert!(lower_layer(&spec.decoder.layers[0]).is_ok());
    if let FeedForwardSpec::Dense { activation, .. } = &mut spec.decoder.layers[0].feed_forward {
        *activation = ActivationSpec::ReluSquared;
    }

    assert_eq!(
        lower_layer(&spec.decoder.layers[0])?.feed_forward,
        FeedForwardLowering::Dense { activation: Activation::ReluSquared }
    );
    assert!(matches!(
        plan(&dense_decoder()?, &spec),
        Err(Error::Architecture(ArchitectureError::NoRuntime { .. }))
    ));
    Ok(())
}

fn dense_spec() -> Result<SemanticModelSpec> {
    let decoder = dense_decoder()?;
    let catalog = TensorCatalog {
        tensors: vec![TensorInfo {
            name: "model.layers.0.input_layernorm.weight".into(),
            file: std::path::PathBuf::new(),
            dtype: "BF16".into(),
            shape: vec![8],
            data_start: 0,
            data_offsets: [0, 0],
        }],
    };
    Ok(SemanticModelSpec::discover(&decoder, &catalog)?)
}

fn dense_decoder() -> Result<DecoderConfig> {
    Ok(DecoderConfig::from_value(&json!({
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "hidden_act": "silu",
        "model_type": "misleading"
    }))?)
}
