use super::HybridMoeLayerConfig;
use crate::engine::{Array, ModelTensors, NormWeight, QuantizedLinear, Result, Stream};

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: QuantizedLinear,
    pub(super) key: QuantizedLinear,
    pub(super) value: Option<QuantizedLinear>,
    pub(super) output: QuantizedLinear,
    pub(super) query_norm: NormWeight,
    pub(super) key_norm: NormWeight,
    pub(super) rope_frequencies: Option<Array>,
}

#[derive(Debug)]
pub(super) struct DenseWeights {
    pub(super) gate: QuantizedLinear,
    pub(super) up: QuantizedLinear,
    pub(super) down: QuantizedLinear,
}

#[derive(Debug)]
pub(super) struct RouterWeights {
    pub(super) projection: QuantizedLinear,
    pub(super) norm_scale: Array,
    pub(super) expert_scale: Array,
}

#[derive(Debug)]
pub(super) struct ExpertWeights {
    pub(super) gate: QuantizedLinear,
    pub(super) up: QuantizedLinear,
    pub(super) down: QuantizedLinear,
}

#[derive(Debug)]
pub(super) struct LayerWeights {
    pub(super) input_norm: NormWeight,
    pub(super) post_attention_norm: NormWeight,
    pub(super) pre_dense_norm: NormWeight,
    pub(super) post_dense_norm: NormWeight,
    pub(super) pre_expert_norm: NormWeight,
    pub(super) post_expert_norm: NormWeight,
    pub(super) post_feed_forward_norm: NormWeight,
    pub(super) layer_scalar: Array,
    pub(super) attention: AttentionWeights,
    pub(super) dense: DenseWeights,
    pub(super) router: RouterWeights,
    pub(super) experts: ExpertWeights,
}

impl LayerWeights {
    pub(super) fn load(
        tensors: &ModelTensors,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let layer = format!("language_model.model.layers.{}", config.layer_index);
        let attention = format!("{layer}.self_attn");
        let dense = format!("{layer}.mlp");
        let router = format!("{layer}.router");
        let experts = format!("{layer}.experts.switch_glu");
        Ok(Self {
            input_norm: NormWeight::load(tensors, &format!("{layer}.input_layernorm"))?,
            post_attention_norm: NormWeight::load(
                tensors,
                &format!("{layer}.post_attention_layernorm"),
            )?,
            pre_dense_norm: NormWeight::load(
                tensors,
                &format!("{layer}.pre_feedforward_layernorm"),
            )?,
            post_dense_norm: NormWeight::load(
                tensors,
                &format!("{layer}.post_feedforward_layernorm_1"),
            )?,
            pre_expert_norm: NormWeight::load(
                tensors,
                &format!("{layer}.pre_feedforward_layernorm_2"),
            )?,
            post_expert_norm: NormWeight::load(
                tensors,
                &format!("{layer}.post_feedforward_layernorm_2"),
            )?,
            post_feed_forward_norm: NormWeight::load(
                tensors,
                &format!("{layer}.post_feedforward_layernorm"),
            )?,
            layer_scalar: tensors.get(&format!("{layer}.layer_scalar"))?,
            attention: load_attention(tensors, &attention, config, stream)?,
            dense: load_dense(tensors, &dense, config.group_size)?,
            router: load_router(tensors, &router, config, stream)?,
            experts: load_experts(tensors, &experts, config.group_size)?,
        })
    }
}

fn load_attention(
    tensors: &ModelTensors,
    prefix: &str,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<AttentionWeights> {
    let rope_frequencies = if config.proportional_rope {
        Some(rope_frequencies(config, stream)?)
    } else {
        None
    };
    Ok(AttentionWeights {
        query: linear(tensors, prefix, "q_proj", config.group_size)?,
        key: linear(tensors, prefix, "k_proj", config.group_size)?,
        value: if config.use_k_eq_v {
            None
        } else {
            Some(linear(tensors, prefix, "v_proj", config.group_size)?)
        },
        output: linear(tensors, prefix, "o_proj", config.group_size)?,
        query_norm: NormWeight::load(tensors, &format!("{prefix}.q_norm"))?,
        key_norm: NormWeight::load(tensors, &format!("{prefix}.k_norm"))?,
        rope_frequencies,
    })
}

fn rope_frequencies(config: HybridMoeLayerConfig, stream: &Stream) -> Result<Array> {
    Array::proportional_rope_frequencies(
        config.head_dim,
        config.rope_dimensions,
        config.rope_base,
        stream,
    )
}

fn load_dense(tensors: &ModelTensors, prefix: &str, group_size: i32) -> Result<DenseWeights> {
    Ok(DenseWeights {
        gate: linear(tensors, prefix, "gate_proj", group_size)?,
        up: linear(tensors, prefix, "up_proj", group_size)?,
        down: linear(tensors, prefix, "down_proj", group_size)?,
    })
}

fn load_router(
    tensors: &ModelTensors,
    prefix: &str,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<RouterWeights> {
    let norm_scale = tensors
        .get(&format!("{prefix}.scale"))?
        .multiply_scalar(config.router_norm_scale, stream)?;
    Ok(RouterWeights {
        projection: linear(tensors, prefix, "proj", config.group_size)?,
        norm_scale,
        expert_scale: tensors.get(&format!("{prefix}.per_expert_scale"))?,
    })
}

fn load_experts(tensors: &ModelTensors, prefix: &str, group_size: i32) -> Result<ExpertWeights> {
    Ok(ExpertWeights {
        gate: linear(tensors, prefix, "gate_proj", group_size)?,
        up: linear(tensors, prefix, "up_proj", group_size)?,
        down: linear(tensors, prefix, "down_proj", group_size)?,
    })
}

fn linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group_size: i32,
) -> Result<QuantizedLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size)
}
