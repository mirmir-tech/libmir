use models::semantic::{ActivationSpec, FeedForwardSpec, MixerSpec, QkNormalization};

use super::fixtures::{AnyResult, dense_and_routed, dense_contract};
use crate::{cuda, metal};

#[test]
fn rejects_unimplemented_norm_offset_scale_and_value_normalization() -> AnyResult<()> {
    let (task, spec) = dense_contract()?;
    // Dense applies an embedding factor and Gemma's unit norm offset (3), but
    // no other offset.
    let mut embedding = spec.clone();
    embedding.decoder.embedding_multiplier = 2.0;
    assert!(metal::admit(&task, Some(&embedding)).is_ok());
    let mut offset = spec.clone();
    offset.decoder.layers[0].input_norm.weight_offset = 1.0;
    offset.decoder.final_norm.weight_offset = 1.0;
    let plan = metal::plan(&offset)?;
    assert_eq!(plan.layers()[0].input_norm, metal::NormalizationLowering::RmsUnitOffset);
    assert_eq!(plan.final_norm(), metal::NormalizationLowering::RmsUnitOffset);
    assert!(metal::admit(&task, Some(&offset)).is_ok());
    offset.decoder.layers[0].input_norm.weight_offset = 0.5;
    assert!(
        metal::admit(&task, Some(&offset))
            .is_err_and(|error| error.to_string().contains("weight offset"))
    );
    let mut scale = spec.clone();
    if let MixerSpec::SoftmaxAttention(attention) = &mut scale.decoder.layers[0].mixer {
        attention.scale = 0.75;
    }
    assert!(
        metal::admit(&task, Some(&scale))
            .is_err_and(|error| error.to_string().contains("attention scale"))
    );
    let mut normalization = spec;
    if let MixerSpec::SoftmaxAttention(attention) = &mut normalization.decoder.layers[0].mixer {
        attention.qk_normalization = QkNormalization::QueryKeyValueRms;
    }
    assert!(
        metal::admit(&task, Some(&normalization))
            .is_err_and(|error| error.to_string().contains("normalization"))
    );
    Ok(())
}

#[test]
fn routed_gemma_accepts_defaults_and_rejects_ignored_overrides() -> AnyResult<()> {
    let (task, dense) = dense_contract()?;
    let mut spec = dense_and_routed(dense);
    assert!(metal::admit(&task, Some(&spec)).is_ok());
    if let MixerSpec::SoftmaxAttention(attention) = &mut spec.decoder.layers[0].mixer {
        attention.qk_normalization = QkNormalization::QueryKeyRms;
    }
    assert!(metal::admit(&task, Some(&spec)).is_err());
    Ok(())
}

#[test]
fn both_backends_refuse_an_activation_known_only_by_name() -> AnyResult<()> {
    let (task, mut semantic) = dense_contract()?;
    for layer in &mut semantic.decoder.layers {
        if let FeedForwardSpec::Dense { activation, .. } = &mut layer.feed_forward {
            *activation = ActivationSpec::NamedGated { name: "quick_gelu".into() };
        }
    }
    assert!(cuda::admit(&task, Some(&semantic)).is_err());
    assert!(metal::admit(&task, Some(&semantic)).is_err());
    Ok(())
}
