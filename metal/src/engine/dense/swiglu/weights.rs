use models::weights::DenseDecoderLayerBindings;

use super::{config::DenseSwiGluLayerConfig, projection::BoundLinear};
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

#[derive(Debug)]
pub(super) struct DenseSwiGluWeights {
    pub(super) input_norm: NormWeight,
    pub(super) post_attention_norm: NormWeight,
    pub(super) attention: AttentionWeights,
    pub(super) mlp: MlpWeights,
}

impl DenseSwiGluWeights {
    pub(super) fn load_bindings(
        tensors: &ModelTensors,
        bindings: DenseDecoderLayerBindings<'_>,
        config: DenseSwiGluLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let norm = |binding: Option<&models::weights::TensorBinding>| {
            binding
                .map(|binding| NormWeight::load_name(tensors, &binding.source))
                .transpose()
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
            mlp: MlpWeights {
                gate: BoundLinear::load(tensors, bindings.gate, stream)?,
                up: BoundLinear::load(tensors, bindings.up, stream)?,
                down: BoundLinear::load(tensors, bindings.down, stream)?,
            },
        })
    }
}

fn rope_frequencies(config: DenseSwiGluLayerConfig, stream: &Stream) -> Result<Option<Array>> {
    config
        .rope_scaling
        .map(|scaling| match scaling {
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
            ),
            models::layout::RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_len,
                ..
            } => Array::yarn_rope_frequencies(
                config.head_dim,
                config.rope_base,
                factor.to_string().parse()?,
                beta_fast.to_string().parse()?,
                beta_slow.to_string().parse()?,
                i32::try_from(original_context_len)?,
                stream,
            ),
        })
        .transpose()
}
