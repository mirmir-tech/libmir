use super::config::DenseSwiGluLayerConfig;
use crate::engine::{Array, ModelTensors, NormWeight, QuantizedLinear, Result, Stream};

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
    pub(super) fn load(
        tensors: &ModelTensors,
        config: DenseSwiGluLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let layer = format!("model.layers.{}", config.index);
        let attention = format!("{layer}.self_attn");
        let mlp = format!("{layer}.mlp");
        Ok(Self {
            input_norm: NormWeight::load(tensors, &format!("{layer}.input_layernorm"))?,
            post_attention_norm: NormWeight::load(
                tensors,
                &format!("{layer}.post_attention_layernorm"),
            )?,
            attention: AttentionWeights {
                query: linear(tensors, &attention, "q_proj", config.group_size)?,
                key: linear(tensors, &attention, "k_proj", config.group_size)?,
                value: linear(tensors, &attention, "v_proj", config.group_size)?,
                output: linear(tensors, &attention, "o_proj", config.group_size)?,
                query_norm: NormWeight::load_optional(tensors, &format!("{attention}.q_norm"))?,
                key_norm: NormWeight::load_optional(tensors, &format!("{attention}.k_norm"))?,
                rope_frequencies: rope_frequencies(config, stream)?,
            },
            mlp: MlpWeights {
                gate: linear(tensors, &mlp, "gate_proj", config.group_size)?,
                up: linear(tensors, &mlp, "up_proj", config.group_size)?,
                down: linear(tensors, &mlp, "down_proj", config.group_size)?,
            },
        })
    }
}

fn rope_frequencies(config: DenseSwiGluLayerConfig, stream: &Stream) -> Result<Option<Array>> {
    config
        .rope_scaling
        .map(|scaling| {
            let (factor, low_frequency_factor, high_frequency_factor, original_context_len) =
                scaling.piecewise_frequency();
            Array::piecewise_rope_frequencies(
                config.head_dim,
                config.rope_base,
                factor.to_string().parse()?,
                low_frequency_factor.to_string().parse()?,
                high_frequency_factor.to_string().parse()?,
                i32::try_from(original_context_len)?,
                stream,
            )
        })
        .transpose()
}

fn linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group_size: i32,
) -> Result<QuantizedLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size)
}
