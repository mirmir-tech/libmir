use models::weights::{
    DenseDecoderLayerBindings, HybridMoeAttentionBindings, HybridMoeDenseBindings,
    HybridMoeExpertBindings, HybridMoeLayerBindings, HybridMoeRouterBindings, TensorBinding,
};

#[cfg(test)]
use self::legacy::{legacy_attention, legacy_dense, legacy_norm, legacy_router};
use super::{
    AttentionWeights, DenseWeights, ExpertGateUpWeights, ExpertWeights, LayerWeights,
    RoutedWeights, RouterWeights,
};
use crate::engine::{
    Array, Error, ModelTensors, NormWeight, Result, Stream, binding::BoundLinear,
    hybrid::moe::HybridMoeLayerConfig,
};

#[cfg(test)]
mod legacy;

impl LayerWeights {
    pub(in crate::engine::hybrid::moe) fn load_bindings(
        tensors: &ModelTensors,
        bindings: &HybridMoeLayerBindings<'_>,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            input_norm: norm(tensors, bindings.input_norm)?,
            post_attention_norm: norm(tensors, bindings.post_attention_norm)?,
            pre_dense_norm: norm(tensors, bindings.pre_dense_norm)?,
            post_feed_forward_norm: norm(tensors, bindings.post_feed_forward_norm)?,
            layer_scalar: tensors.get(&bindings.layer_scale.source)?,
            attention: attention(tensors, bindings.attention, config, stream)?,
            dense: dense(tensors, bindings.dense, stream)?,
            routed: Some(RoutedWeights {
                post_dense_norm: norm(tensors, bindings.post_dense_norm)?,
                pre_expert_norm: norm(tensors, bindings.pre_expert_norm)?,
                post_expert_norm: norm(tensors, bindings.post_expert_norm)?,
                router: router(tensors, bindings.router, config, stream)?,
                experts: experts(tensors, &bindings.experts, stream)?,
            }),
        })
    }

    pub(in crate::engine::hybrid::moe) fn load_dense_bindings(
        tensors: &ModelTensors,
        bindings: &DenseDecoderLayerBindings<'_>,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let sandwich = bindings.sandwich.ok_or_else(|| {
            Error::InvalidModel("dense GELU layer needs output norms and a layer scale".into())
        })?;
        let attention_norm = |binding: Option<&TensorBinding>| {
            binding.cloned().ok_or_else(|| {
                Error::InvalidModel("dense GELU attention needs query and key norms".into())
            })
        };
        let layer_scale = sandwich
            .layer_scale
            .ok_or_else(|| Error::InvalidModel("dense GELU layer needs a layer scale".into()))?;
        let mlp = bindings.feed_forward.gated()?;
        let query_norm = attention_norm(bindings.attention.query_norm)?;
        let key_norm = attention_norm(bindings.attention.key_norm)?;
        Ok(Self {
            input_norm: norm(tensors, bindings.input_norm)?,
            post_attention_norm: norm(tensors, sandwich.attention_output_norm)?,
            pre_dense_norm: norm(tensors, bindings.post_attention_norm)?,
            post_feed_forward_norm: norm(tensors, sandwich.feed_forward_output_norm)?,
            layer_scalar: tensors.get(&layer_scale.source)?,
            attention: attention(
                tensors,
                HybridMoeAttentionBindings {
                    query: bindings.attention.query,
                    key: bindings.attention.key,
                    value: Some(bindings.attention.value),
                    output: bindings.attention.output,
                    query_norm: &query_norm,
                    key_norm: &key_norm,
                },
                config,
                stream,
            )?,
            dense: dense(
                tensors,
                HybridMoeDenseBindings {
                    gate: mlp.gate,
                    up: mlp.up,
                    down: mlp.down,
                },
                stream,
            )?,
            routed: None,
        })
    }

    #[cfg(test)]
    pub(in crate::engine::hybrid::moe) fn load(
        tensors: &ModelTensors,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let layer = format!("language_model.model.layers.{}", config.layer_index);
        Ok(Self {
            input_norm: legacy_norm(tensors, &layer, "input_layernorm")?,
            post_attention_norm: legacy_norm(tensors, &layer, "post_attention_layernorm")?,
            pre_dense_norm: legacy_norm(tensors, &layer, "pre_feedforward_layernorm")?,
            post_feed_forward_norm: legacy_norm(tensors, &layer, "post_feedforward_layernorm")?,
            layer_scalar: tensors.get(&format!("{layer}.layer_scalar"))?,
            attention: legacy_attention(tensors, &format!("{layer}.self_attn"), config, stream)?,
            dense: legacy_dense(tensors, &format!("{layer}.mlp"), config.group_size)?,
            routed: Some(RoutedWeights {
                post_dense_norm: legacy_norm(tensors, &layer, "post_feedforward_layernorm_1")?,
                pre_expert_norm: legacy_norm(tensors, &layer, "pre_feedforward_layernorm_2")?,
                post_expert_norm: legacy_norm(tensors, &layer, "post_feedforward_layernorm_2")?,
                router: legacy_router(tensors, &format!("{layer}.router"), config, stream)?,
                experts: legacy_dense(
                    tensors,
                    &format!("{layer}.experts.switch_glu"),
                    config.group_size,
                )?
                .into(),
            }),
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
        Some(BoundLinear::load(tensors, binding, stream)?)
    };
    Ok(AttentionWeights {
        query: BoundLinear::load(tensors, bindings.query, stream)?,
        key: BoundLinear::load(tensors, bindings.key, stream)?,
        value,
        output: BoundLinear::load(tensors, bindings.output, stream)?,
        query_norm: norm(tensors, bindings.query_norm)?,
        key_norm: norm(tensors, bindings.key_norm)?,
        rope_frequencies: config
            .proportional_rope
            .then(|| rope_frequencies(config, stream))
            .transpose()?,
    })
}

fn dense(
    tensors: &ModelTensors,
    bindings: HybridMoeDenseBindings<'_>,
    stream: &Stream,
) -> Result<DenseWeights> {
    Ok(DenseWeights {
        gate: BoundLinear::load(tensors, bindings.gate, stream)?,
        up: BoundLinear::load(tensors, bindings.up, stream)?,
        down: BoundLinear::load(tensors, bindings.down, stream)?,
    })
}

fn router(
    tensors: &ModelTensors,
    bindings: HybridMoeRouterBindings<'_>,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<RouterWeights> {
    Ok(RouterWeights {
        projection: BoundLinear::load(tensors, bindings.projection, stream)?,
        norm_scale: tensors
            .get(&bindings.norm_scale.source)?
            .multiply_scalar(config.router_norm_scale, stream)?,
        expert_scale: tensors.get(&bindings.expert_scale.source)?,
    })
}

fn experts(
    tensors: &ModelTensors,
    bindings: &HybridMoeExpertBindings<'_>,
    stream: &Stream,
) -> Result<ExpertWeights> {
    match bindings {
        HybridMoeExpertBindings::Stacked(bindings) => {
            let weights = dense(tensors, *bindings, stream)?;
            Ok(weights.into())
        },
        HybridMoeExpertBindings::FusedStacked { gate_up, down } => {
            let output = gate_up.shape.get(1).copied().ok_or(Error::ShapeOverflow)?;
            if !output.is_multiple_of(2) {
                return Err(Error::InvalidModel("fused expert gate/up width must be even".into()));
            }
            Ok(ExpertWeights {
                gate_up: ExpertGateUpWeights::Fused {
                    projection: BoundLinear::load(tensors, gate_up, stream)?,
                    width: output / 2,
                    interleaved: gate_up.transforms.contains(
                        &models::weights::BindingTransform::FusedGateUp { interleaved: true },
                    ),
                },
                down: BoundLinear::load(tensors, down, stream)?,
            })
        },
        HybridMoeExpertBindings::Individual { gate, up, down } => Ok(ExpertWeights {
            gate_up: ExpertGateUpWeights::Separate {
                gate: BoundLinear::load_nvfp4_bank(tensors, gate, stream)?,
                up: BoundLinear::load_nvfp4_bank(tensors, up, stream)?,
            },
            down: BoundLinear::load_nvfp4_bank(tensors, down, stream)?,
        }),
    }
}

fn norm(tensors: &ModelTensors, binding: &TensorBinding) -> Result<NormWeight> {
    NormWeight::load_name(tensors, &binding.source)
}

pub(super) fn rope_frequencies(config: HybridMoeLayerConfig, stream: &Stream) -> Result<Array> {
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
            gate_up: ExpertGateUpWeights::Separate { gate: weights.gate, up: weights.up },
            down: weights.down,
        }
    }
}
