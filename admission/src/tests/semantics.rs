use models::semantic::{ActivationSpec, FeedForwardSpec, MixerSpec, QkNormalization};

use super::fixtures::{AnyResult, dense_and_routed, dense_contract};
use crate::{
    ArchitectureError, Constant, Runtime, Unlowerable,
    capability::{AttentionScale, AttentionScaleValue, Requirement, ScaleSource},
    cuda,
    metal::{self, MetalDecoderRuntime},
};

/// Whether `runtime` refused the plan for lacking `requirement`.
fn lacks(error: Option<ArchitectureError>, runtime: Runtime, requirement: Requirement) -> bool {
    let Some(ArchitectureError::NoRuntime { rejections }) = error else {
        return false;
    };
    rejections
        .iter()
        .any(|rejection| rejection.runtime == runtime && rejection.missing.contains(&requirement))
}

#[test]
fn rejects_unimplemented_norm_offset_scale_and_value_normalization() -> AnyResult<()> {
    let (task, spec) = dense_contract()?;
    let dense = Runtime::Metal(MetalDecoderRuntime::Dense);
    // Dense applies an embedding factor and Gemma's unit norm offset, but no
    // other offset.
    let mut embedding = spec.clone();
    embedding.decoder.embedding_multiplier = 2.0;
    assert!(metal::admit(&task, Some(&embedding)).is_ok());
    let mut offset = spec.clone();
    offset.decoder.layers[0].input_norm.weight_offset = 1.0;
    offset.decoder.final_norm.weight_offset = 1.0;
    let models::execution::TaskExecutionPlan::Generation { decoder } = &task else {
        return Err("the fixture generates".into());
    };
    let plan = metal::plan(decoder, &offset)?;
    assert_eq!(plan.layers()[0].input_norm, metal::NormalizationLowering::RmsUnitOffset);
    assert_eq!(plan.final_norm(), metal::NormalizationLowering::RmsUnitOffset);
    assert_eq!(plan.runtime(), MetalDecoderRuntime::Dense);
    offset.decoder.layers[0].input_norm.weight_offset = 0.5;
    assert_eq!(
        metal::admit(&task, Some(&offset)).err(),
        Some(ArchitectureError::Unlowerable {
            layer: Some(0),
            feature: Unlowerable::NormWeightOffset { offset: Constant(0.5) },
        })
    );
    let mut scale = spec.clone();
    if let MixerSpec::SoftmaxAttention(attention) = &mut scale.decoder.layers[0].mixer {
        attention.scale = 0.75;
    }
    let undeclared = Requirement::AttentionScale(AttentionScale {
        value: AttentionScaleValue::Other,
        source: ScaleSource::Override,
    });
    assert!(lacks(metal::admit(&task, Some(&scale)).err(), dense, undeclared));
    let mut normalization = spec;
    if let MixerSpec::SoftmaxAttention(attention) = &mut normalization.decoder.layers[0].mixer {
        attention.qk_normalization = QkNormalization::QueryKeyValueRms;
    }
    let value = Requirement::Normalization(QkNormalization::QueryKeyValueRms);
    assert!(lacks(metal::admit(&task, Some(&normalization)).err(), dense, value));
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
    let routed = Runtime::Metal(MetalDecoderRuntime::DenseAndRouted);
    let keys = Requirement::Normalization(QkNormalization::QueryKeyRms);
    assert!(lacks(metal::admit(&task, Some(&spec)).err(), routed, keys));
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
    let refusal = Some(ArchitectureError::Unlowerable {
        layer: Some(0),
        feature: Unlowerable::Activation { name: "quick_gelu".into() },
    });
    assert_eq!(cuda::admit(&task, Some(&semantic)).err(), refusal);
    assert_eq!(metal::admit(&task, Some(&semantic)).err(), refusal);
    Ok(())
}
