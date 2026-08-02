use super::*;

#[test]
fn discovers_individual_shared_routed_experts() -> Result<()> {
    let decoder = decoder()?;
    let mut catalog = catalog();
    catalog.tensors.retain(|tensor| !tensor.name.contains(".mlp.switch_mlp."));
    for layer in 0..2 {
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
        for expert in 0..8 {
            dense(
                &mut catalog.tensors,
                &format!("{prefix}.{expert}.gate_proj.weight"),
                &[16, HIDDEN],
            );
            dense(
                &mut catalog.tensors,
                &format!("{prefix}.{expert}.up_proj.weight"),
                &[16, HIDDEN],
            );
            dense(
                &mut catalog.tensors,
                &format!("{prefix}.{expert}.down_proj.weight"),
                &[HIDDEN, 16],
            );
        }
    }
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    let plan = WeightBindingPlan::discover(&spec, &catalog)?;

    let RoutedExpertBindings::Individual { .. } = plan.hybrid_decoder_layer(0)?.feed_forward.routed
    else {
        return Err(invalid("expected individual routed expert projections"));
    };
    assert_eq!(
        plan.hybrid_decoder_layer(0)?
            .feed_forward
            .routed
            .individual(ExpertProjectionRole::Gate)
            .len(),
        8
    );
    assert_eq!(individual_count(&plan), 48);
    Ok(())
}

fn individual_count(plan: &WeightBindingPlan) -> usize {
    plan.tensors
        .iter()
        .filter(|binding| {
            matches!(
                binding.role,
                LogicalTensorRole::Layer {
                    tensor: LayerTensorRole::ExpertProjection { expert: Some(_), .. },
                    ..
                }
            )
        })
        .count()
}
