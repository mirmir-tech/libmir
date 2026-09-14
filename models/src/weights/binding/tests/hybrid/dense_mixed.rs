use super::*;
use crate::weights::MixedFeedForwardBindings;

#[test]
fn discovers_dense_mixed_attention_and_rejects_missing_projection() -> Result<()> {
    let mut decoder = decoder()?;
    decoder.num_experts = None;
    decoder.top_k_experts = None;
    decoder.moe_intermediate_size = None;
    decoder.shared_expert_intermediate_size = None;
    decoder.intermediate_size = 64;
    let mut catalog = catalog();
    catalog.tensors.retain(|tensor| !tensor.name.contains(".mlp."));
    for index in 0..2 {
        let prefix = format!("language_model.model.layers.{index}.mlp");
        for name in ["gate_proj", "up_proj"] {
            dense(&mut catalog.tensors, &format!("{prefix}.{name}.weight"), &[64, HIDDEN]);
        }
        dense(&mut catalog.tensors, &format!("{prefix}.down_proj.weight"), &[HIDDEN, 64]);
    }
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    for index in 0..2 {
        let view = plan.mixed_decoder_layer(index)?;
        assert!(matches!(view.feed_forward, MixedFeedForwardBindings::Dense(_)));
        assert!(
            view.physical_sources()
                .iter()
                .any(|name| name.ends_with("mlp.down_proj.weight"))
        );
        assert!(plan.hybrid_decoder_layer(index).is_err());
    }
    catalog.tensors.retain(|tensor| !tensor.name.ends_with("0.mlp.up_proj.weight"));
    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}
