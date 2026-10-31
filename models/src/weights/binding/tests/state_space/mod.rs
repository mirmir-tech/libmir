use serde_json::json;

use super::*;
use crate::semantic::{FeedForwardSpec, MixerSpec, PositionEncodingSpec};

mod fixture;
mod prediction;

use fixture::{EXPERTS, HIDDEN, block, catalog, config, dense, sorted};

fn plan(catalog: &TensorCatalog) -> Result<(SemanticModelSpec, WeightBindingPlan)> {
    let decoder = DecoderConfig::from_value(&config())?;
    let spec = SemanticModelSpec::discover(&decoder, catalog)?;
    let plan = WeightBindingPlan::discover(&spec, catalog)?;
    Ok((spec, plan))
}

fn rejects(catalog: &TensorCatalog, needle: &str) -> Result<()> {
    rejected(plan(catalog).map(|_| ()), needle)
}

fn rejected(result: Result<()>, needle: &str) -> Result<()> {
    match result {
        Ok(()) => Err(invalid("checkpoint must be rejected")),
        Err(error) => {
            assert!(error.to_string().contains(needle), "unexpected error: {error}");
            Ok(())
        },
    }
}

fn invalid(message: &str) -> crate::ModelsError {
    crate::ModelsError::InvalidConfig(message.into())
}

#[test]
fn composes_paired_semantic_layers() -> Result<()> {
    let (spec, _) = plan(&catalog())?;
    let layers = &spec.decoder.layers;

    assert_eq!(layers.len(), 3);
    assert!(matches!(layers[0].mixer, MixerSpec::StateSpace(_)));
    assert!(matches!(layers[0].feed_forward, FeedForwardSpec::LatentRouted { .. }));
    assert_eq!(layers[1].feed_forward, FeedForwardSpec::None);
    let MixerSpec::SoftmaxAttention(attention) = &layers[2].mixer else {
        return Err(invalid("layer 2 must be attention"));
    };
    assert_eq!(attention.position, PositionEncodingSpec::None);
    let toml = spec.to_toml()?;
    assert!(toml.contains("kind = \"paired\""), "{toml}");
    Ok(())
}

#[test]
fn binds_every_hybrid_layer_through_its_typed_view() -> Result<()> {
    let (_, plan) = plan(&catalog())?;

    let first = plan.state_space_hybrid_layer(0)?;
    let StateSpaceHybridMixerBindings::StateSpace(mamba) = first.mixer else {
        return Err(invalid("layer 0 must be a state-space mixer"));
    };
    assert_eq!(mamba.input_projection.source, format!("{}.mixer.in_proj.weight", block(0)));
    assert!(matches!(mamba.convolution.storage, TensorStorage::Dense { bias: Some(_), .. }));
    let routed = first.feed_forward.ok_or_else(|| invalid("layer 0 has a routed block"))?;
    assert_eq!(routed.norm.source, format!("{}.norm.weight", block(1)));
    assert_eq!(routed.routed.individual(ExpertProjectionRole::Up).len(), EXPERTS);
    assert!(routed.routed.individual(ExpertProjectionRole::Gate).is_empty());
    assert!(routed.shared.is_some());

    assert!(plan.state_space_hybrid_layer(1)?.feed_forward.is_none());
    let last = plan.state_space_hybrid_layer(2)?;
    assert!(matches!(last.mixer, StateSpaceHybridMixerBindings::Attention(_)));
    assert_eq!(last.input_norm.source, format!("{}.norm.weight", block(3)));
    Ok(())
}

#[test]
fn reads_rank_zero_and_rank_one_tensor_scales_alike() -> Result<()> {
    let (_, plan) = plan(&catalog())?;
    let mamba = match plan.state_space_hybrid_layer(0)?.mixer {
        StateSpaceHybridMixerBindings::StateSpace(mamba) => mamba,
        StateSpaceHybridMixerBindings::Attention(_) => {
            return Err(invalid("layer 0 must be a state-space mixer"));
        },
    };
    for binding in [mamba.input_projection, mamba.output_projection] {
        let TensorStorage::Float8 { format, .. } = &binding.storage else {
            return Err(invalid("state-space projections must use direct FP8 storage"));
        };
        assert_eq!(format.scale_granularity, Float8ScaleGranularity::Tensor);
        assert_eq!(format.activation_scale, Float8ActivationScale::StaticTensor);
    }
    assert!(plan.uses_block_format(BlockFormat::NvFp4));
    Ok(())
}

#[test]
fn leaves_multimodal_and_prediction_tensors_unbound_to_layers() -> Result<()> {
    let (_, plan) = plan(&catalog())?;
    for path in ["mtp.layers.0.eh_proj.weight", "vision_model.radio_model.patch.weight"] {
        let role = LogicalTensorRole::Auxiliary { path: path.to_owned() };
        assert!(plan.binding(&role).is_some(), "{path} must stay auxiliary");
    }
    Ok(())
}

#[test]
fn rejects_a_missing_router_correction_bias() -> Result<()> {
    let mut catalog = catalog();
    catalog
        .tensors
        .retain(|tensor| !tensor.name.ends_with("4.mixer.gate.e_score_correction_bias"));
    rejects(&catalog, "incomplete binding grammar")
}

#[test]
fn rejects_tensors_outside_their_block_kind() -> Result<()> {
    let mut tensors = catalog().tensors;
    dense(&mut tensors, &format!("{}.mixer.gate.weight", block(0)), &[EXPERTS, HIDDEN]);
    rejects(&sorted(tensors), "has no role in its block")?;

    let mut tensors = catalog().tensors;
    dense(&mut tensors, &format!("{}.mixer.mystery.weight", block(2)), &[HIDDEN]);
    rejects(&sorted(tensors), "has no role in its block")
}

#[test]
fn rejects_a_block_without_tensors() -> Result<()> {
    let mut catalog = catalog();
    let prefix = format!("{}.", block(2));
    catalog.tensors.retain(|tensor| !tensor.name.starts_with(&prefix));
    rejects(&catalog, "no tensors for block 2")
}

#[test]
fn rejects_a_group_count_that_disagrees_with_the_projections() -> Result<()> {
    let mut config = config();
    config["llm_config"]["n_groups"] = json!(4);
    let decoder = DecoderConfig::from_value(&config)?;
    let catalog = catalog();
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    rejected(WeightBindingPlan::discover(&spec, &catalog).map(|_| ()), "expected shape")
}
