use models::weights::{
    HybridMoeAttentionBindings, HybridMoeDenseBindings, HybridMoeExpertBindings,
    HybridMoeLayerBindings, HybridMoeRouterBindings, TensorBinding,
};

use super::{AttentionWeights, DenseWeights, ExpertWeights, LayerWeights, RouterWeights};
#[cfg(test)]
use crate::engine::QuantizedLinear;
use crate::engine::{
    Array, Error, ModelTensors, NormWeight, Result, Stream, binding::affine_linear,
    hybrid_moe::HybridMoeLayerConfig,
};

impl LayerWeights {
    pub(in crate::engine::hybrid_moe) fn load_bindings(
        tensors: &ModelTensors,
        bindings: &HybridMoeLayerBindings<'_>,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            input_norm: norm(tensors, bindings.input_norm)?,
            post_attention_norm: norm(tensors, bindings.post_attention_norm)?,
            pre_dense_norm: norm(tensors, bindings.pre_dense_norm)?,
            post_dense_norm: norm(tensors, bindings.post_dense_norm)?,
            pre_expert_norm: norm(tensors, bindings.pre_expert_norm)?,
            post_expert_norm: norm(tensors, bindings.post_expert_norm)?,
            post_feed_forward_norm: norm(tensors, bindings.post_feed_forward_norm)?,
            layer_scalar: tensors.get(&bindings.layer_scale.source)?,
            attention: attention(tensors, bindings.attention, config, stream)?,
            dense: dense(tensors, bindings.dense)?,
            router: router(tensors, bindings.router, config, stream)?,
            experts: experts(tensors, &bindings.experts)?,
        })
    }

    #[cfg(test)]
    pub(in crate::engine::hybrid_moe) fn load(
        tensors: &ModelTensors,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let layer = format!("language_model.model.layers.{}", config.layer_index);
        Ok(Self {
            input_norm: legacy_norm(tensors, &layer, "input_layernorm")?,
            post_attention_norm: legacy_norm(tensors, &layer, "post_attention_layernorm")?,
            pre_dense_norm: legacy_norm(tensors, &layer, "pre_feedforward_layernorm")?,
            post_dense_norm: legacy_norm(tensors, &layer, "post_feedforward_layernorm_1")?,
            pre_expert_norm: legacy_norm(tensors, &layer, "pre_feedforward_layernorm_2")?,
            post_expert_norm: legacy_norm(tensors, &layer, "post_feedforward_layernorm_2")?,
            post_feed_forward_norm: legacy_norm(tensors, &layer, "post_feedforward_layernorm")?,
            layer_scalar: tensors.get(&format!("{layer}.layer_scalar"))?,
            attention: legacy_attention(tensors, &format!("{layer}.self_attn"), config, stream)?,
            dense: legacy_dense(tensors, &format!("{layer}.mlp"), config.group_size)?,
            router: legacy_router(tensors, &format!("{layer}.router"), config, stream)?,
            experts: legacy_dense(
                tensors,
                &format!("{layer}.experts.switch_glu"),
                config.group_size,
            )?
            .into(),
        })
    }
}

fn attention(
    tensors: &ModelTensors,
    bindings: HybridMoeAttentionBindings<'_>,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<AttentionWeights> {
    let value = if config.use_k_eq_v {
        None
    } else {
        let binding = bindings.value.ok_or_else(|| {
            Error::InvalidModel("hybrid MoE attention value projection is unbound".into())
        })?;
        Some(affine_linear(tensors, binding)?)
    };
    Ok(AttentionWeights {
        query: affine_linear(tensors, bindings.query)?,
        key: affine_linear(tensors, bindings.key)?,
        value,
        output: affine_linear(tensors, bindings.output)?,
        query_norm: norm(tensors, bindings.query_norm)?,
        key_norm: norm(tensors, bindings.key_norm)?,
        rope_frequencies: config
            .proportional_rope
            .then(|| rope_frequencies(config, stream))
            .transpose()?,
    })
}

fn dense(tensors: &ModelTensors, bindings: HybridMoeDenseBindings<'_>) -> Result<DenseWeights> {
    Ok(DenseWeights {
        gate: affine_linear(tensors, bindings.gate)?,
        up: affine_linear(tensors, bindings.up)?,
        down: affine_linear(tensors, bindings.down)?,
    })
}

fn router(
    tensors: &ModelTensors,
    bindings: HybridMoeRouterBindings<'_>,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<RouterWeights> {
    Ok(RouterWeights {
        projection: affine_linear(tensors, bindings.projection)?,
        norm_scale: tensors
            .get(&bindings.norm_scale.source)?
            .multiply_scalar(config.router_norm_scale, stream)?,
        expert_scale: tensors.get(&bindings.expert_scale.source)?,
    })
}

fn experts(
    tensors: &ModelTensors,
    bindings: &HybridMoeExpertBindings<'_>,
) -> Result<ExpertWeights> {
    let HybridMoeExpertBindings::Stacked(bindings) = bindings else {
        return Err(Error::InvalidModel(
            "Metal hybrid MoE requires stacked expert bindings".into(),
        ));
    };
    let weights = dense(tensors, *bindings)?;
    Ok(weights.into())
}

fn norm(tensors: &ModelTensors, binding: &TensorBinding) -> Result<NormWeight> {
    NormWeight::load_name(tensors, &binding.source)
}

fn rope_frequencies(config: HybridMoeLayerConfig, stream: &Stream) -> Result<Array> {
    Array::proportional_rope_frequencies(
        config.head_dim,
        config.rope_dimensions,
        config.rope_base,
        stream,
    )
}

impl From<DenseWeights> for ExpertWeights {
    fn from(weights: DenseWeights) -> Self {
        Self {
            gate: weights.gate,
            up: weights.up,
            down: weights.down,
        }
    }
}

#[cfg(test)]
fn legacy_norm(tensors: &ModelTensors, prefix: &str, name: &str) -> Result<NormWeight> {
    NormWeight::load(tensors, &format!("{prefix}.{name}"))
}

#[cfg(test)]
fn legacy_attention(
    tensors: &ModelTensors,
    prefix: &str,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<AttentionWeights> {
    Ok(AttentionWeights {
        query: legacy_linear(tensors, prefix, "q_proj", config.group_size)?,
        key: legacy_linear(tensors, prefix, "k_proj", config.group_size)?,
        value: (!config.use_k_eq_v)
            .then(|| legacy_linear(tensors, prefix, "v_proj", config.group_size))
            .transpose()?,
        output: legacy_linear(tensors, prefix, "o_proj", config.group_size)?,
        query_norm: legacy_norm(tensors, prefix, "q_norm")?,
        key_norm: legacy_norm(tensors, prefix, "k_norm")?,
        rope_frequencies: config
            .proportional_rope
            .then(|| rope_frequencies(config, stream))
            .transpose()?,
    })
}

#[cfg(test)]
fn legacy_dense(tensors: &ModelTensors, prefix: &str, group: i32) -> Result<DenseWeights> {
    Ok(DenseWeights {
        gate: legacy_linear(tensors, prefix, "gate_proj", group)?,
        up: legacy_linear(tensors, prefix, "up_proj", group)?,
        down: legacy_linear(tensors, prefix, "down_proj", group)?,
    })
}

#[cfg(test)]
fn legacy_router(
    tensors: &ModelTensors,
    prefix: &str,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<RouterWeights> {
    Ok(RouterWeights {
        projection: legacy_linear(tensors, prefix, "proj", config.group_size)?,
        norm_scale: tensors
            .get(&format!("{prefix}.scale"))?
            .multiply_scalar(config.router_norm_scale, stream)?,
        expert_scale: tensors.get(&format!("{prefix}.per_expert_scale"))?,
    })
}

#[cfg(test)]
fn legacy_linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group: i32,
) -> Result<QuantizedLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group)
}
