use models::semantic::{MixerSpec, QkNormalization};

use super::fixtures::{AnyResult, dense_and_routed, dense_contract};
use crate::metal;

#[test]
fn rejects_unimplemented_embedding_scale_and_value_normalization() -> AnyResult<()> {
    let (task, spec) = dense_contract()?;
    let mut embedding = spec.clone();
    embedding.decoder.embedding_multiplier = 2.0;
    assert!(
        metal::admit(&task, Some(&embedding))
            .is_err_and(|error| error.to_string().contains("embedding_multiplier"))
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
