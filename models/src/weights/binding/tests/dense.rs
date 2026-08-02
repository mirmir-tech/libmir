use super::*;

#[test]
fn resolves_dense_layer_without_reconstructing_checkpoint_names() -> Result<()> {
    let plan = plan();

    let layer = plan.dense_decoder_layer(0)?;

    assert_eq!(layer.attention.query.source, "custom.query.kernel");
    assert_eq!(
        layer.attention.query_norm.map(|binding| binding.source.as_str()),
        Some("custom.query.scale")
    );
    assert!(layer.attention.key_norm.is_none());
    assert!(layer.physical_sources().contains(&"custom.gate.parameters"));
    Ok(())
}

#[test]
fn resolves_tied_output_from_the_embedding_role() -> Result<()> {
    let plan = plan();

    let boundary = plan.decoder_boundary_with_tied_output(true)?;

    assert_eq!(boundary.embedding.source, "custom.token_table");
    assert_eq!(boundary.output.source, boundary.embedding.source);
    Ok(())
}

#[test]
fn accepts_pre_feed_forward_norm_as_the_dense_residual_norm() -> Result<()> {
    let mut plan = plan();
    let binding = plan
        .tensors
        .iter_mut()
        .find(|binding| {
            binding.role
                == LogicalTensorRole::Layer {
                    index: 0,
                    tensor: LayerTensorRole::PostAttentionNorm,
                }
        })
        .ok_or_else(|| crate::ModelsError::InvalidConfig("missing test norm".into()))?;
    binding.role = LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::PreDenseNorm,
    };

    let layer = plan.dense_decoder_layer(0)?;

    assert_eq!(layer.post_attention_norm.source, "custom.post.scale");
    Ok(())
}

#[test]
fn rejects_incomplete_dense_layer_views() {
    let mut plan = plan();
    plan.tensors.retain(|binding| {
        binding.role
            != LogicalTensorRole::Layer {
                index: 0,
                tensor: LayerTensorRole::FeedForwardProjection {
                    projection: FeedForwardProjectionRole::Down,
                },
            }
    });

    assert!(plan.dense_decoder_layer(0).is_err());
}

fn plan() -> WeightBindingPlan {
    WeightBindingPlan {
        tensors: vec![
            dense(LogicalTensorRole::Embedding, "custom.token_table"),
            dense(LogicalTensorRole::FinalNorm, "custom.final.scale"),
            layer(LayerTensorRole::InputNorm, "custom.input.scale"),
            attention(AttentionProjectionRole::Query, "custom.query.kernel"),
            attention(AttentionProjectionRole::Key, "custom.key.kernel"),
            attention(AttentionProjectionRole::Value, "custom.value.kernel"),
            attention(AttentionProjectionRole::Output, "custom.output.kernel"),
            layer(LayerTensorRole::QueryNorm, "custom.query.scale"),
            layer(LayerTensorRole::PostAttentionNorm, "custom.post.scale"),
            affine(FeedForwardProjectionRole::Gate, "custom.gate.kernel", "custom.gate.parameters"),
            feed_forward(FeedForwardProjectionRole::Up, "custom.up.kernel"),
            feed_forward(FeedForwardProjectionRole::Down, "custom.down.kernel"),
        ],
    }
}

fn attention(projection: AttentionProjectionRole, source: &str) -> TensorBinding {
    layer(LayerTensorRole::AttentionProjection { projection }, source)
}

fn feed_forward(projection: FeedForwardProjectionRole, source: &str) -> TensorBinding {
    layer(LayerTensorRole::FeedForwardProjection { projection }, source)
}

fn layer(tensor: LayerTensorRole, source: &str) -> TensorBinding {
    dense(LogicalTensorRole::Layer { index: 0, tensor }, source)
}

fn affine(projection: FeedForwardProjectionRole, source: &str, scales: &str) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::FeedForwardProjection { projection },
        },
        source: source.into(),
        shape: Vec::new(),
        logical_shape: None,
        transforms: Vec::new(),
        storage: TensorStorage::AffineQuantized {
            format: GroupedAffineQuantization {
                bits: AffineBits::Four,
                group_size: 32,
                group_axis: AffineGroupAxis::Input,
                signedness: AffineSignedness::Unsigned,
                zero_point: AffineZeroPointMode::None,
                packing: AffinePacking::Mlx,
                storage_dtype: AffineStorageDType::U32,
                scale_dtype: AffineParameterDType::F16,
                bias_dtype: None,
            },
            scales: scales.into(),
            biases: None,
            output_bias: None,
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
