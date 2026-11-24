use models::{
    execution::TaskExecutionPlan,
    layout::DecoderConfig,
    semantic::{
        ActivationSpec, AttentionOutputSpec, FeedForwardSpec, LinearAttentionSpec, MixerSpec,
        NormalizationKind, SemanticModelSpec,
    },
    weights::{TensorCatalog, TensorInfo},
};
use serde_json::json;

use super::*;

#[test]
fn materializes_dense_layers_without_model_identity() -> Result<()> {
    let spec = dense_spec()?;
    let lowering = plan(&spec)?;

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
                window: None
            },
            feed_forward: FeedForwardLowering::Dense,
        }]
    );
    assert_eq!(lowering.runtime(), DecoderRuntime::Dense);
    assert_eq!(
        crate::admit_architecture(
            &TaskExecutionPlan::Generation { decoder: dense_decoder()? },
            Some(&spec),
        )?,
        crate::MetalArchitecture::Generation(DecoderRuntime::Dense)
    );
    Ok(())
}

#[test]
fn rejects_an_unavailable_normalization_independently() -> Result<()> {
    let mut spec = dense_spec()?;
    spec.decoder.layers[0].input_norm.kind = NormalizationKind::Layer;
    spec.decoder.layers[0].input_norm.weight_offset = 1.0;

    let error = lower_layer(&spec.decoder.layers[0])
        .err()
        .ok_or_else(|| Error::InvalidModel("offset layer normalization was admitted".into()))?;

    assert!(error.to_string().contains("input layer normalization"));
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
    assert_eq!(plan(&spec)?.runtime(), DecoderRuntime::DenseMixed);
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
                .map(models::semantic::AttentionWindow::Sliding)
        }
    );
    // The dense runtime runs sliding layers (Gemma 3) with the cache's mask.
    assert_eq!(plan(&spec)?.runtime(), DecoderRuntime::Dense);
    Ok(())
}

#[test]
fn rejects_an_activation_the_dense_operator_does_not_implement() -> Result<()> {
    let mut spec = dense_spec()?;
    // Tanh GELU (Gemma's GeGLU) lowers; an ungated activation lowers to the
    // hybrid MLP, which no softmax-only runtime executes.
    if let FeedForwardSpec::Dense { activation, .. } = &mut spec.decoder.layers[0].feed_forward {
        *activation = ActivationSpec::GeluTanh;
    }
    assert!(lower_layer(&spec.decoder.layers[0]).is_ok());
    if let FeedForwardSpec::Dense { activation, .. } = &mut spec.decoder.layers[0].feed_forward {
        *activation = ActivationSpec::ReluSquared;
    }

    assert_eq!(
        lower_layer(&spec.decoder.layers[0])?.feed_forward,
        FeedForwardLowering::UngatedDense
    );
    let error = plan(&spec)
        .err()
        .ok_or_else(|| Error::InvalidModel("dense relu-squared was admitted".into()))?;
    assert!(error.to_string().contains("no runtime"));
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
