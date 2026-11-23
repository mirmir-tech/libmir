use models::{
    layout::DecoderConfig,
    semantic::QkNormalization,
    weights::{
        DenseDecoderLayerBindings, DenseLayerFeedForwardBindings, DenseSoftmaxBindings,
        ParallelDecoderLayerBindings, PostNormDecoderLayerBindings, TensorBinding,
    },
};

use super::{
    super::{config::DenseSwiGluLayerConfig, projection::BoundLinear, routed::RoutedMlp},
    AttentionWeights, DenseSwiGluWeights, FeedForwardWeights, LayerNorms, MlpWeights, QkNormScope,
    SandwichNorms, rope_frequencies,
};
use crate::engine::{
    Error, ModelTensors, NormWeight, Result, SharedExpertMoe, Stream,
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
        decoder: &DecoderConfig,
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
            attention: attention(tensors, bindings.attention, config, normalization, stream)?,
            feed_forward: feed_forward(tensors, decoder, bindings.feed_forward, config, stream)?,
        })
    }

    /// A post-norm block (`OLMo` 2): norms on each sub-block's output only.
    pub(in crate::engine::dense::swiglu) fn load_post_norm(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
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
            attention: attention(tensors, bindings.attention, config, normalization, stream)?,
            feed_forward: feed_forward(tensors, decoder, bindings.feed_forward, config, stream)?,
        })
    }
}

impl DenseSwiGluWeights {
    /// A parallel block (Cohere): one norm feeds both sub-blocks.
    pub(in crate::engine::dense::swiglu) fn load_parallel(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        bindings: ParallelDecoderLayerBindings<'_>,
        config: DenseSwiGluLayerConfig,
        normalization: QkNormalization,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            norms: LayerNorms::Parallel {
                input: layer_norm(tensors, bindings.input_norm, config, stream)?,
            },
            attention: attention(tensors, bindings.attention, config, normalization, stream)?,
            feed_forward: feed_forward(tensors, decoder, bindings.feed_forward, config, stream)?,
        })
    }
}

fn attention(
    tensors: &ModelTensors,
    bindings: DenseSoftmaxBindings<'_>,
    config: DenseSwiGluLayerConfig,
    normalization: QkNormalization,
    stream: &Stream,
) -> Result<AttentionWeights> {
    let norm = |binding: Option<&TensorBinding>| {
        match normalization {
            // Unit norms have no weight; the attention applies them.
            QkNormalization::None | QkNormalization::QueryKeyUnitRms => return Ok(None),
            QkNormalization::QueryKeyValueRms => {
                return Err(Error::InvalidModel(
                    "dense Metal attention does not support value RMS normalization".into(),
                ));
            },
            QkNormalization::QueryKeyRms | QkNormalization::QueryKeyRmsAcrossHeads
                if binding.is_none() =>
            {
                return Err(Error::InvalidModel(
                    "declared Q/K RMS normalization is missing its weight".into(),
                ));
            },
            QkNormalization::QueryKeyRms | QkNormalization::QueryKeyRmsAcrossHeads => {},
        }
        binding
            .map(|binding| {
                NormWeight::load_name_adjusted(
                    tensors,
                    &binding.source,
                    config.norm_shift(),
                    stream,
                )
            })
            .transpose()
    };
    Ok(AttentionWeights {
        query: BoundLinear::load(tensors, bindings.query, stream)?,
        key: BoundLinear::load(tensors, bindings.key, stream)?,
        value: BoundLinear::load(tensors, bindings.value, stream)?,
        output: BoundLinear::load(tensors, bindings.output, stream)?,
        query_norm: norm(bindings.query_norm)?,
        key_norm: norm(bindings.key_norm)?,
        norm_scope: match normalization {
            QkNormalization::QueryKeyRmsAcrossHeads => QkNormScope::Projection,
            QkNormalization::QueryKeyUnitRms => QkNormScope::UnitHead,
            QkNormalization::None
            | QkNormalization::QueryKeyRms
            | QkNormalization::QueryKeyValueRms => QkNormScope::Head,
        },
        rope_frequencies: rope_frequencies(config, stream)?,
    })
}

fn feed_forward(
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    bindings: DenseLayerFeedForwardBindings<'_>,
    config: DenseSwiGluLayerConfig,
    stream: &Stream,
) -> Result<FeedForwardWeights> {
    Ok(match bindings {
        DenseLayerFeedForwardBindings::Gated(bindings) => FeedForwardWeights::Gated(MlpWeights {
            gate: BoundLinear::load(tensors, bindings.gate, stream)?,
            up: BoundLinear::load(tensors, bindings.up, stream)?,
            down: BoundLinear::load(tensors, bindings.down, stream)?,
        }),
        DenseLayerFeedForwardBindings::Routed(bindings) => {
            let top_k = config
                .top_k_experts
                .ok_or_else(|| Error::InvalidModel("routed experts need a top-k".into()))?;
            FeedForwardWeights::Routed(RoutedMlp::load(
                tensors,
                bindings,
                top_k,
                config.sparse_mixer_jitter,
                stream,
            )?)
        },
        DenseLayerFeedForwardBindings::SharedRouted(bindings) => {
            FeedForwardWeights::SharedRouted(Box::new(SharedExpertMoe::load_decoder(
                tensors,
                decoder,
                bindings,
                FeedForwardLowering::SharedRouted,
                stream,
            )?))
        },
    })
}
