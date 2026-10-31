//! Multi-token prediction modules bound beside the hybrid decoder.

use super::{
    fixture::{EXPERTS, HIDDEN, catalog, dense, prediction_catalog, prediction_config, sorted},
    rejected,
};
use crate::{
    error::Result,
    layout::DecoderConfig,
    semantic::{FeedForwardSpec, MixerSpec, SemanticModelSpec},
    weights::{
        AttentionProjectionRole, LogicalTensorRole, PredictionTensorRole,
        StateSpaceHybridMixerBindings, TensorCatalog, WeightBindingPlan,
    },
};

fn plan(catalog: &TensorCatalog) -> Result<(SemanticModelSpec, WeightBindingPlan)> {
    let decoder = DecoderConfig::from_value(&prediction_config())?;
    let spec = SemanticModelSpec::discover(&decoder, catalog)?;
    let plan = WeightBindingPlan::discover(&spec, catalog)?;
    Ok((spec, plan))
}

#[test]
fn binds_one_prediction_module_with_decoder_layer_roles() -> Result<()> {
    let (spec, plan) = plan(&prediction_catalog())?;
    let [module] = spec.decoder.prediction_layers.as_slice() else {
        return Err(crate::ModelsError::InvalidConfig("one prediction module".into()));
    };
    assert!(matches!(module.mixer, MixerSpec::SoftmaxAttention(_)));
    assert!(matches!(module.feed_forward, FeedForwardSpec::LatentRouted { .. }));

    let bindings = plan.prediction_layer(0)?;
    assert_eq!(bindings.fusion.source, "language_model.mtp.layers.0.eh_proj.weight");
    assert_eq!(bindings.layer.input_norm.source, "language_model.mtp.layers.0.norm.weight");
    assert!(matches!(bindings.layer.mixer, StateSpaceHybridMixerBindings::Attention(_)));
    let routed = bindings.layer.feed_forward.ok_or_else(|| {
        crate::ModelsError::InvalidConfig("prediction module needs its routed block".into())
    })?;
    assert_eq!(routed.norm.source, "language_model.mtp.layers.1.norm.weight");
    assert_eq!(
        routed.routed.individual(crate::weights::ExpertProjectionRole::Up).len(),
        EXPERTS
    );
    assert_eq!(bindings.final_norm.source, "language_model.mtp.layers.1.final_layernorm.weight");
    let scale = LogicalTensorRole::Prediction {
        depth: 0,
        tensor: PredictionTensorRole::CacheScale { projection: AttentionProjectionRole::Key },
    };
    assert!(plan.binding(&scale).is_some());
    // Expert companions and the four module tensors are all read.
    assert!(bindings.physical_sources().len() > 4 * EXPERTS);
    Ok(())
}

#[test]
fn keeps_prediction_tensors_auxiliary_without_declared_modules() -> Result<()> {
    let decoder = DecoderConfig::from_value(&super::fixture::config())?;
    let catalog = prediction_catalog();
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    assert!(spec.decoder.prediction_layers.is_empty());
    assert!(plan.prediction_layer(0).is_err());
    let role = LogicalTensorRole::Auxiliary {
        path: "mtp.layers.1.final_layernorm.weight".into(),
    };
    assert!(plan.binding(&role).is_some());
    Ok(())
}

#[test]
fn rejects_incomplete_or_unknown_prediction_tensors() -> Result<()> {
    let mut missing = prediction_catalog();
    missing
        .tensors
        .retain(|tensor| !tensor.name.ends_with("mtp.layers.0.hnorm.weight"));
    rejected(
        plan(&missing).map(|_| ()),
        "prediction module 0 has an incomplete binding grammar",
    )?;

    let mut tensors = prediction_catalog().tensors;
    dense(&mut tensors, "language_model.mtp.layers.0.mixer.mystery.weight", &[HIDDEN]);
    rejected(plan(&sorted(tensors)).map(|_| ()), "has no role in its module")?;

    let mut tensors = prediction_catalog().tensors;
    dense(&mut tensors, "language_model.mtp.layers.2.enorm.weight", &[HIDDEN]);
    rejected(plan(&sorted(tensors)).map(|_| ()), "has no role in its module")?;

    // The decoder-only catalog has just `eh_proj` of the declared module.
    rejected(plan(&catalog()).map(|_| ()), "incomplete binding grammar")
}
