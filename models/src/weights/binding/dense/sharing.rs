//! Bindings of Gemma 4's E models: layers that attend over an earlier
//! layer's keys and values, and the per-layer inputs.

use super::{
    super::{PerLayerEmbeddingRole, PerLayerInputRole},
    DenseLayerFeedForwardBindings, DenseSandwichBindings, LayerTensorRole, LogicalTensorRole,
    TensorBinding, WeightBindingPlan, feed_forward, layer, norms, optional,
};
use crate::error::{ModelsError, Result};

/// A dense layer whose attention reads an earlier layer's keys and values:
/// it projects queries only, and checkpoints may drop its unused key and
/// value tensors.
#[derive(Debug, Clone, Copy)]
pub struct SharedKeyValueLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub query: &'a TensorBinding,
    pub output: &'a TensorBinding,
    pub query_norm: Option<&'a TensorBinding>,
    pub post_attention_norm: &'a TensorBinding,
    pub sandwich: Option<DenseSandwichBindings<'a>>,
    pub feed_forward: DenseLayerFeedForwardBindings<'a>,
}

/// The model's tensors forming every layer's input.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbeddingBindings<'a> {
    pub embedding: &'a TensorBinding,
    pub model_projection: &'a TensorBinding,
    pub projection_norm: &'a TensorBinding,
}

/// A layer's tensors gating its per-layer input into its output.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerInputBindings<'a> {
    pub gate: &'a TensorBinding,
    pub projection: &'a TensorBinding,
    pub output_norm: &'a TensorBinding,
}

impl WeightBindingPlan {
    pub fn dense_decoder_layer_sharing_key_values(
        &self,
        index: usize,
    ) -> Result<SharedKeyValueLayerBindings<'_>> {
        let (sandwich, post_attention_norm) = norms(self, index)?;
        let projection =
            |projection| layer(self, index, LayerTensorRole::AttentionProjection { projection });
        Ok(SharedKeyValueLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            query: projection(super::AttentionProjectionRole::Query)?,
            output: projection(super::AttentionProjectionRole::Output)?,
            query_norm: optional(self, index, LayerTensorRole::QueryNorm),
            post_attention_norm,
            sandwich,
            feed_forward: feed_forward(self, index)?,
        })
    }

    /// The per-layer input tensors, when the checkpoint has them.
    pub fn per_layer_embedding(&self) -> Result<Option<PerLayerEmbeddingBindings<'_>>> {
        let tensor = |tensor| self.binding(&LogicalTensorRole::PerLayerEmbedding { tensor });
        let roles = [
            PerLayerEmbeddingRole::Embedding,
            PerLayerEmbeddingRole::ModelProjection,
            PerLayerEmbeddingRole::ProjectionNorm,
        ];
        match roles.map(tensor) {
            [None, None, None] => Ok(None),
            [Some(embedding), Some(model_projection), Some(projection_norm)] => {
                Ok(Some(PerLayerEmbeddingBindings {
                    embedding,
                    model_projection,
                    projection_norm,
                }))
            },
            _ => Err(ModelsError::InvalidConfig("per-layer input tensors are incomplete".into())),
        }
    }

    /// Layer `index`'s per-layer input gate, when the checkpoint has one.
    pub fn per_layer_input(&self, index: usize) -> Result<Option<PerLayerInputBindings<'_>>> {
        let tensor = |tensor| optional(self, index, LayerTensorRole::PerLayerInput { tensor });
        let roles =
            [PerLayerInputRole::Gate, PerLayerInputRole::Projection, PerLayerInputRole::OutputNorm];
        match roles.map(tensor) {
            [None, None, None] => Ok(None),
            [Some(gate), Some(projection), Some(output_norm)] => {
                Ok(Some(PerLayerInputBindings { gate, projection, output_norm }))
            },
            _ => Err(ModelsError::InvalidConfig(format!(
                "layer {index}'s per-layer input tensors are incomplete"
            ))),
        }
    }
}
