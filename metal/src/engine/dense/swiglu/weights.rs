use models::{
    semantic::QkNormalization,
    weights::{DenseDecoderLayerBindings, DenseLayerFeedForwardBindings},
};

use super::{config::DenseSwiGluLayerConfig, projection::BoundLinear, routed::RoutedMlp};
use crate::engine::{Array, ModelTensors, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: BoundLinear,
    pub(super) key: BoundLinear,
    pub(super) value: BoundLinear,
    pub(super) output: BoundLinear,
    pub(super) query_norm: Option<NormWeight>,
    pub(super) key_norm: Option<NormWeight>,
    pub(super) rope_frequencies: Option<Array>,
}

#[derive(Debug)]
pub(super) struct MlpWeights {
    pub(super) gate: BoundLinear,
    pub(super) up: BoundLinear,
    pub(super) down: BoundLinear,
}

/// The layer's feed-forward: a gated MLP, or routed experts in its place.
#[derive(Debug)]
pub(super) enum FeedForwardWeights {
    Gated(MlpWeights),
    Routed(RoutedMlp),
}

/// Gemma-style norms on each sub-block's output before its residual add.
#[derive(Debug)]
pub(super) struct SandwichNorms {
    pub(super) attention_output: NormWeight,
    pub(super) feed_forward_output: NormWeight,
}

#[derive(Debug)]
pub(super) struct DenseSwiGluWeights {
    pub(super) input_norm: NormWeight,
    pub(super) post_attention_norm: NormWeight,
    pub(super) sandwich: Option<SandwichNorms>,
    pub(super) attention: AttentionWeights,
    pub(super) feed_forward: FeedForwardWeights,
}

impl DenseSwiGluWeights {
    pub(super) fn load_bindings(
        tensors: &ModelTensors,
        bindings: DenseDecoderLayerBindings<'_>,
        config: DenseSwiGluLayerConfig,
        normalization: QkNormalization,
        stream: &Stream,
    ) -> Result<Self> {
        let norm = |binding: Option<&models::weights::TensorBinding>| {
            match normalization {
                QkNormalization::None => return Ok(None),
                QkNormalization::QueryKeyValueRms => {
                    return Err(crate::engine::Error::InvalidModel(
                        "dense Metal attention does not support value RMS normalization".into(),
                    ));
                },
                QkNormalization::QueryKeyRms if binding.is_none() => {
                    return Err(crate::engine::Error::InvalidModel(
                        "declared Q/K RMS normalization is missing its weight".into(),
                    ));
                },
                QkNormalization::QueryKeyRms => {},
            }
            binding
                .map(|binding| NormWeight::load_name(tensors, &binding.source))
                .transpose()
        };
        let feed_forward = match bindings.feed_forward {
            DenseLayerFeedForwardBindings::Gated(bindings) => {
                FeedForwardWeights::Gated(MlpWeights {
                    gate: BoundLinear::load(tensors, bindings.gate, stream)?,
                    up: BoundLinear::load(tensors, bindings.up, stream)?,
                    down: BoundLinear::load(tensors, bindings.down, stream)?,
                })
            },
            DenseLayerFeedForwardBindings::Routed(bindings) => {
                let top_k = config.top_k_experts.ok_or_else(|| {
                    crate::engine::Error::InvalidModel("routed experts need a top-k".into())
                })?;
                FeedForwardWeights::Routed(RoutedMlp::load(tensors, bindings, top_k, stream)?)
            },
        };
        Ok(Self {
            input_norm: NormWeight::load_name(tensors, &bindings.input_norm.source)?,
            post_attention_norm: NormWeight::load_name(
                tensors,
                &bindings.post_attention_norm.source,
            )?,
            attention: AttentionWeights {
                query: BoundLinear::load(tensors, bindings.attention.query, stream)?,
                key: BoundLinear::load(tensors, bindings.attention.key, stream)?,
                value: BoundLinear::load(tensors, bindings.attention.value, stream)?,
                output: BoundLinear::load(tensors, bindings.attention.output, stream)?,
                query_norm: norm(bindings.attention.query_norm)?,
                key_norm: norm(bindings.attention.key_norm)?,
                rope_frequencies: rope_frequencies(config, stream)?,
            },
            sandwich: bindings
                .sandwich
                .map(|sandwich| {
                    if sandwich.layer_scale.is_some() {
                        return Err(crate::engine::Error::InvalidModel(
                            "dense Metal layers have no learned layer scale".into(),
                        ));
                    }
                    Ok(SandwichNorms {
                        attention_output: NormWeight::load_name(
                            tensors,
                            &sandwich.attention_output_norm.source,
                        )?,
                        feed_forward_output: NormWeight::load_name(
                            tensors,
                            &sandwich.feed_forward_output_norm.source,
                        )?,
                    })
                })
                .transpose()?,
            feed_forward,
        })
    }
}

fn rope_frequencies(config: DenseSwiGluLayerConfig, stream: &Stream) -> Result<Option<Array>> {
    config
        .rope_scaling
        .map(|scaling| match scaling {
            // Linear scaling divides positions: `rope_position_scale`.
            models::layout::RopeScaling::Linear { .. } => Ok(None),
            models::layout::RopeScaling::PiecewiseFrequency {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_context_len,
            } => Array::piecewise_rope_frequencies(
                config.head_dim,
                config.rope_base,
                factor.to_string().parse()?,
                low_frequency_factor.to_string().parse()?,
                high_frequency_factor.to_string().parse()?,
                i32::try_from(original_context_len)?,
                stream,
            )
            .map(Some),
            models::layout::RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
                truncate,
                ..
            } => Array::yarn_rope_frequencies(
                config.head_dim,
                config.rope_base,
                factor.to_string().parse()?,
                beta_fast.to_string().parse()?,
                beta_slow.to_string().parse()?,
                i32::try_from(original_context_len)?,
                truncate,
                stream,
            )
            .map(Some),
        })
        .transpose()
        .map(Option::flatten)
}
