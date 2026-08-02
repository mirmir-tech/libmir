use super::*;

#[test]
fn resolves_a_routed_layer_without_reconstructing_checkpoint_names() -> Result<()> {
    let plan = plan(true);

    let layer = plan.routed_decoder_layer(0)?;

    assert_eq!(layer.query.source, "nested.decoder.layers.0.attention.query.kernel");
    assert!(matches!(layer.experts, RoutedExpertBindings::InterleavedGateUp { .. }));
    assert!(layer.physical_sources().contains(&"custom.gate_up.scale"));
    assert!(layer.physical_sources().contains(&"custom.gate_up.bias"));
    Ok(())
}

#[test]
fn resolves_separate_expert_projections_by_role() -> Result<()> {
    let plan = plan(false);

    let layer = plan.routed_decoder_layer(0)?;

    let RoutedExpertBindings::SeparateGateUp { gate, up, down } = layer.experts else {
        return Err(crate::ModelsError::InvalidConfig("expected separate expert view".into()));
    };
    assert_eq!(gate.source, "custom.gate.weight");
    assert_eq!(up.source, "custom.up.weight");
    assert_eq!(down.source, "custom.down.weight");
    Ok(())
}

#[test]
fn resolves_decoder_boundary_by_semantic_role() -> Result<()> {
    let plan = plan(true);

    let boundary = plan.decoder_boundary()?;

    assert_eq!(boundary.embedding.source, "custom.token_table");
    assert_eq!(boundary.final_norm.source, "custom.final_scale");
    assert_eq!(boundary.output.source, "custom.logits.kernel");
    Ok(())
}

fn plan(interleaved: bool) -> WeightBindingPlan {
    let mut tensors = vec![
        dense(LogicalTensorRole::Embedding, "custom.token_table"),
        dense(LogicalTensorRole::FinalNorm, "custom.final_scale"),
        dense(LogicalTensorRole::Output, "custom.logits.kernel"),
        layer(LayerTensorRole::InputNorm, "custom.input_scale"),
        attention(AttentionProjectionRole::Query, "nested.decoder.layers.0.attention.query.kernel"),
        attention(AttentionProjectionRole::Key, "custom.key.kernel"),
        attention(AttentionProjectionRole::Value, "custom.value.kernel"),
        attention(AttentionProjectionRole::Output, "custom.attention_output.kernel"),
        layer(LayerTensorRole::AttentionSinks, "custom.sinks"),
        layer(LayerTensorRole::PostAttentionNorm, "custom.post_scale"),
        layer(LayerTensorRole::Router, "custom.router.kernel"),
        block_expert(ExpertProjectionRole::Down, "custom.down.weight"),
    ];
    if interleaved {
        tensors.push(block_expert(ExpertProjectionRole::GateUp, "custom.gate_up.weight"));
    } else {
        tensors.extend([
            block_expert(ExpertProjectionRole::Gate, "custom.gate.weight"),
            block_expert(ExpertProjectionRole::Up, "custom.up.weight"),
        ]);
    }
    WeightBindingPlan { tensors }
}

fn attention(projection: AttentionProjectionRole, source: &str) -> TensorBinding {
    layer(LayerTensorRole::AttentionProjection { projection }, source)
}

fn layer(tensor: LayerTensorRole, source: &str) -> TensorBinding {
    dense(LogicalTensorRole::Layer { index: 0, tensor }, source)
}

fn block_expert(projection: ExpertProjectionRole, source: &str) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
        },
        source: source.into(),
        shape: Vec::new(),
        logical_shape: None,
        transforms: Vec::new(),
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: if projection == ExpertProjectionRole::GateUp {
                "custom.gate_up.scale".into()
            } else {
                format!("{source}.scale")
            },
            global_scale: None,
            input_scale: None,
            bias: Some(if projection == ExpertProjectionRole::GateUp {
                "custom.gate_up.bias".into()
            } else {
                format!("{source}.bias")
            }),
            packing: TensorPacking::Separate,
        },
    }
}

fn dense(role: LogicalTensorRole, source: &str) -> TensorBinding {
    TensorBinding {
        role,
        source: source.into(),
        shape: Vec::new(),
        logical_shape: None,
        transforms: Vec::new(),
        storage: TensorStorage::Dense { dtype: "BF16".into(), bias: None },
    }
}
