//! A Gemma dense layer attending over an earlier layer's keys and values
//! (Gemma 4's E models): it loads its query and output projections only.

use std::sync::Arc;

use models::weights::{HybridMoeDenseBindings, SharedKeyValueLayerBindings};

use super::{
    super::{super::sharing::StepShare, AttentionWeights, KeyValueWeights, LayerWeights},
    dense, norm,
};
use crate::engine::{Array, Error, ModelTensors, Result, Stream, binding::BoundLinear};

impl LayerWeights {
    pub(in crate::engine::hybrid::moe) fn load_shared_dense_bindings(
        tensors: &ModelTensors,
        bindings: &SharedKeyValueLayerBindings<'_>,
        rope_frequencies: Option<Array>,
        (step, source): (Arc<StepShare>, usize),
        stream: &Stream,
    ) -> Result<Self> {
        let sandwich = bindings.sandwich.ok_or_else(|| {
            Error::InvalidModel("dense GELU layer needs output norms and a layer scale".into())
        })?;
        let layer_scale = sandwich
            .layer_scale
            .ok_or_else(|| Error::InvalidModel("dense GELU layer needs a layer scale".into()))?;
        let query_norm = bindings
            .query_norm
            .ok_or_else(|| Error::InvalidModel("dense GELU attention needs a query norm".into()))?;
        let mlp = bindings.feed_forward.gated()?;
        Ok(Self {
            input_norm: norm(tensors, bindings.input_norm)?,
            post_attention_norm: norm(tensors, sandwich.attention_output_norm)?,
            pre_dense_norm: norm(tensors, bindings.post_attention_norm)?,
            post_feed_forward_norm: norm(tensors, sandwich.feed_forward_output_norm)?,
            layer_scalar: tensors.get(&layer_scale.source)?,
            attention: AttentionWeights {
                query: BoundLinear::load(tensors, bindings.query, stream)?,
                key_values: KeyValueWeights::Shared { step, source },
                output: BoundLinear::load(tensors, bindings.output, stream)?,
                query_norm: norm(tensors, query_norm)?,
                rope_frequencies,
            },
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
}
