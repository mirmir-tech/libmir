use models::{
    semantic::{DecoderLayerSpec, QkNormalization},
    weights::{
        DenseDecoderLayerBindings, DenseLayerFeedForwardBindings, DenseSoftmaxBindings,
        ParallelDecoderLayerBindings, PostNormDecoderLayerBindings, TensorBinding,
    },
};

use super::{
    super::{config::DenseSwiGluLayerConfig, projection::BoundLinear},
    AttentionWeights, DenseSwiGluWeights, FeedForwardWeights, LayerNorms, MlpWeights, QkNorm,
    SandwichNorms,
};
use crate::engine::{
    Array, Error, ModelTensors, NormWeight, Result, SharedExpertMoe, Stream,
    lowering::{FeedForwardLowering, NormalizationLowering},
};

/// A layer norm as the layer's lowering scales it.
fn layer_norm(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    config: DenseSwiGluLayerConfig,
    stream: &Stream,
) -> Result<NormWeight> {
    let norm =
        NormWeight::load_name_adjusted(tensors, &binding.source, config.norm_shift(), stream)?;
    match config.norm {
        NormalizationLowering::Layer => {
            norm.into_layer(binding.dense_bias().map(|bias| tensors.get(bias)).transpose()?, stream)
        },
        NormalizationLowering::Rms | NormalizationLowering::RmsUnitOffset => Ok(norm),
    }
}

impl DenseSwiGluWeights {
    pub(in crate::engine::dense::swiglu) fn load_bindings(
        tensors: &ModelTensors,
        layer: &DecoderLayerSpec,
        bindings: DenseDecoderLayerBindings<'_>,
        config: DenseSwiGluLayerConfig,
        normalization: QkNormalization,
        stream: &Stream,
    ) -> Result<Self> {
        // Gemma's `1 + weight` applies to every norm of the layer.
        let norm = |binding: &TensorBinding| layer_norm(tensors, binding, config, stream);
        let output = bindings
            .sandwich
            .map(|sandwich| {
                if sandwich.layer_scale.is_some() {
                    return Err(Error::InvalidModel(
                        "dense Metal layers have no learned layer scale".into(),
                    ));
                }
                Ok(SandwichNorms {
                    attention_output: norm(sandwich.attention_output_norm)?,
                    feed_forward_output: norm(sandwich.feed_forward_output_norm)?,
                })
            })
            .transpose()?;
        Ok(Self {
            norms: LayerNorms::Pre {
                input: norm(bindings.input_norm)?,
                feed_forward: norm(bindings.post_attention_norm)?,
                output,
            },
            attention: attention(
                tensors,
                layer,
                bindings.attention,
                config,
                normalization,
                stream,
            )?,
            feed_forward: feed_forward(tensors, layer, bindings.feed_forward, stream)?,
        })
    }

    /// A post-norm block (`OLMo` 2): norms on each sub-block's output only.
    pub(in crate::engine::dense::swiglu) fn load_post_norm(
        tensors: &ModelTensors,
        layer: &DecoderLayerSpec,
        bindings: PostNormDecoderLayerBindings<'_>,
        config: DenseSwiGluLayerConfig,
        normalization: QkNormalization,
        stream: &Stream,
    ) -> Result<Self> {
        let norm = |binding: &TensorBinding| layer_norm(tensors, binding, config, stream);
        Ok(Self {
            norms: LayerNorms::Post(SandwichNorms {
                attention_output: norm(bindings.attention_output_norm)?,
                feed_forward_output: norm(bindings.feed_forward_output_norm)?,
            }),
            attention: attention(
                tensors,
                layer,
                bindings.attention,
                config,
                normalization,
                stream,
            )?,
            feed_forward: feed_forward(tensors, layer, bindings.feed_forward, stream)?,
        })
    }
}

impl DenseSwiGluWeights {
    /// A parallel block (Cohere): one norm feeds both sub-blocks.
    pub(in crate::engine::dense::swiglu) fn load_parallel(
        tensors: &ModelTensors,
        layer: &DecoderLayerSpec,
        bindings: ParallelDecoderLayerBindings<'_>,
        config: DenseSwiGluLayerConfig,
        normalization: QkNormalization,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            norms: LayerNorms::Parallel {
                input: layer_norm(tensors, bindings.input_norm, config, stream)?,
            },
            attention: attention(
                tensors,
                layer,
                bindings.attention,
                config,
                normalization,
                stream,
            )?,
            feed_forward: feed_forward(tensors, layer, bindings.feed_forward, stream)?,
        })
    }
}

fn attention(
    tensors: &ModelTensors,
    layer: &DecoderLayerSpec,
    bindings: DenseSoftmaxBindings<'_>,
    config: DenseSwiGluLayerConfig,
    normalization: QkNormalization,
    stream: &Stream,
) -> Result<AttentionWeights> {
    let norm = |binding: Option<&TensorBinding>| {
        let binding = binding.ok_or_else(|| {
            Error::InvalidModel("declared Q/K RMS normalization is missing its weight".into())
        })?;
        NormWeight::load_name_adjusted(tensors, &binding.source, config.norm_shift(), stream)
    };
    let qk_norm = match normalization {
        QkNormalization::None => QkNorm::None,
        // Unit norms have no weight; the attention applies them.
        QkNormalization::QueryKeyUnitRms => QkNorm::UnitHead,
        QkNormalization::QueryKeyRms => QkNorm::Head {
            query: norm(bindings.query_norm)?,
            key: norm(bindings.key_norm)?,
        },
        QkNormalization::QueryKeyRmsAcrossHeads => QkNorm::Projection {
            query: norm(bindings.query_norm)?,
            key: norm(bindings.key_norm)?,
        },
        QkNormalization::QueryKeyValueRms => {
            return Err(Error::InvalidModel(
                "dense Metal attention does not support value RMS normalization".into(),
            ));
        },
    };
    Ok(AttentionWeights {
        query: BoundLinear::load(tensors, bindings.query, stream)?,
        key: BoundLinear::load(tensors, bindings.key, stream)?,
        value: BoundLinear::load(tensors, bindings.value, stream)?,
        output: BoundLinear::load(tensors, bindings.output, stream)?,
        qk_norm,
        rope_frequencies: Array::rope_periods(super::super::config::attention(layer)?)?,
    })
}

fn feed_forward(
    tensors: &ModelTensors,
    layer: &DecoderLayerSpec,
    bindings: DenseLayerFeedForwardBindings<'_>,
    stream: &Stream,
) -> Result<FeedForwardWeights> {
    Ok(match bindings {
        DenseLayerFeedForwardBindings::Gated(bindings) => {
            FeedForwardWeights::Gated(Box::new(MlpWeights {
                gate: BoundLinear::load(tensors, bindings.gate, stream)?,
                up: BoundLinear::load(tensors, bindings.up, stream)?,
                down: BoundLinear::load(tensors, bindings.down, stream)?,
            }))
        },
        DenseLayerFeedForwardBindings::Routed(bindings) => FeedForwardWeights::Routed(Box::new(
            SharedExpertMoe::load_routed_spec(tensors, routed(layer)?, bindings, stream)?,
        )),
        DenseLayerFeedForwardBindings::SharedRouted(bindings) => {
            FeedForwardWeights::Routed(Box::new(SharedExpertMoe::load_spec(
                tensors,
                routed(layer)?,
                bindings,
                FeedForwardLowering::SharedRouted,
                stream,
            )?))
        },
    })
}

fn routed(layer: &DecoderLayerSpec) -> Result<&models::semantic::RoutedExpertsSpec> {
    layer
        .feed_forward
        .routed()
        .ok_or_else(|| Error::InvalidModel("routed bindings need routed experts".into()))
}
