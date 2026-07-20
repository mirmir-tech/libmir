use models::weights::TextTensorLayout;

use super::LayerConfig;
use crate::engine::{DenseLinear, ModelTensors, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub query: DenseLinear,
    pub key: DenseLinear,
    pub value: DenseLinear,
    pub output: DenseLinear,
    pub query_norm: Option<NormWeight>,
    pub key_norm: Option<NormWeight>,
}

#[derive(Debug)]
pub(super) struct LayerWeights {
    pub input_norm: NormWeight,
    pub post_attention_norm: NormWeight,
    pub attention: AttentionWeights,
    pub gate: DenseLinear,
    pub up: DenseLinear,
    pub down: DenseLinear,
}

impl LayerWeights {
    pub fn load(
        tensors: &ModelTensors,
        config: LayerConfig,
        layout: &TextTensorLayout,
        stream: &Stream,
    ) -> Result<Self> {
        let prefix = layout.name(format!("layers.{}", config.index));
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        Ok(Self {
            input_norm: NormWeight::load(tensors, &format!("{prefix}.input_layernorm"))?,
            post_attention_norm: NormWeight::load(
                tensors,
                &format!("{prefix}.post_attention_layernorm"),
            )?,
            attention: AttentionWeights {
                query: DenseLinear::load(tensors, &format!("{attention}.q_proj"), stream)?,
                key: DenseLinear::load(tensors, &format!("{attention}.k_proj"), stream)?,
                value: DenseLinear::load(tensors, &format!("{attention}.v_proj"), stream)?,
                output: DenseLinear::load(tensors, &format!("{attention}.o_proj"), stream)?,
                query_norm: NormWeight::load_optional(tensors, &format!("{attention}.q_norm"))?,
                key_norm: NormWeight::load_optional(tensors, &format!("{attention}.k_norm"))?,
            },
            gate: DenseLinear::load(tensors, &format!("{mlp}.gate_proj"), stream)?,
            up: DenseLinear::load(tensors, &format!("{mlp}.up_proj"), stream)?,
            down: DenseLinear::load(tensors, &format!("{mlp}.down_proj"), stream)?,
        })
    }
}
