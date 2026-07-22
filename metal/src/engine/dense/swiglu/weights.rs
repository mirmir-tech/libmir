use models::weights::DenseDecoderLayerBindings;

use super::config::DenseSwiGluLayerConfig;
use crate::engine::{
    Array, ModelTensors, NormWeight, QuantizedLinear, Result, Stream, binding::affine_linear,
};

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: QuantizedLinear,
    pub(super) key: QuantizedLinear,
    pub(super) value: QuantizedLinear,
    pub(super) output: QuantizedLinear,
    pub(super) query_norm: Option<NormWeight>,
    pub(super) key_norm: Option<NormWeight>,
    pub(super) rope_frequencies: Option<Array>,
}

#[derive(Debug)]
pub(super) struct MlpWeights {
    pub(super) gate: QuantizedLinear,
    pub(super) up: QuantizedLinear,
    pub(super) down: QuantizedLinear,
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
                query: affine_linear(tensors, bindings.attention.query)?,
                key: affine_linear(tensors, bindings.attention.key)?,
                value: affine_linear(tensors, bindings.attention.value)?,
                output: affine_linear(tensors, bindings.attention.output)?,
                query_norm: norm(bindings.attention.query_norm)?,
                key_norm: norm(bindings.attention.key_norm)?,
                rope_frequencies: rope_frequencies(config, stream)?,
            },
            mlp: MlpWeights {
                gate: affine_linear(tensors, bindings.gate)?,
                up: affine_linear(tensors, bindings.up)?,
                down: affine_linear(tensors, bindings.down)?,
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
