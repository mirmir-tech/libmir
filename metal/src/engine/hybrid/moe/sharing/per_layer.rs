//! Gemma 4's per-layer inputs (MLX-LM `Gemma4TextModel._get_per_layer_inputs`,
//! `_project_per_layer_inputs` and `DecoderLayer`'s per-layer gate).

use std::sync::Arc;

use models::weights::{PerLayerEmbeddingBindings, PerLayerInputBindings};

use super::StepShare;
use crate::engine::{
    Array, ModelTensors, NormWeight, Result, Stream,
    binding::{BoundEmbedding, BoundLinear},
};

/// The model's half: every layer's input from the token ids and the scaled
/// token embeddings.
#[derive(Debug)]
pub(in crate::engine::hybrid::moe) struct PerLayerInputs {
    embedding: BoundEmbedding,
    model_projection: BoundLinear,
    projection_norm: NormWeight,
    layers: i32,
    width: i32,
    /// `sqrt(width)` on the per-layer embedding.
    embedding_scale: f32,
    /// `1 / sqrt(hidden)` on the projection of the token embeddings.
    projection_scale: f32,
    epsilon: f32,
}

impl PerLayerInputs {
    pub(in crate::engine::hybrid::moe) fn load(
        tensors: &ModelTensors,
        bindings: PerLayerEmbeddingBindings<'_>,
        [layers, width, hidden]: [usize; 3],
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let width_f32 = width.to_string().parse::<f32>()?;
        Ok(Self {
            embedding: BoundEmbedding::load(tensors, bindings.embedding, stream)?,
            model_projection: BoundLinear::load(tensors, bindings.model_projection, stream)?,
            projection_norm: NormWeight::load_name(tensors, &bindings.projection_norm.source)?,
            layers: i32::try_from(layers)?,
            width: i32::try_from(width)?,
            embedding_scale: width_f32.sqrt(),
            projection_scale: hidden.to_string().parse::<f32>()?.sqrt().recip(),
            epsilon,
        })
    }

    /// `[B, S, layers, width]` inputs of `token_ids [B, S]`, whose token
    /// embeddings are `scaled [B, S, hidden]` (already times `sqrt(hidden)`).
    pub(in crate::engine::hybrid::moe) fn compute(
        &self,
        token_ids: &Array,
        scaled: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = token_ids.shape()?;
        let split = [shape[0], shape[1], self.layers, self.width];
        let embedded = self
            .embedding
            .lookup(token_ids, stream)?
            .multiply_scalar(self.embedding_scale, stream)?
            .reshape(&split, stream)?;
        let projected = self
            .model_projection
            .forward(scaled, stream)?
            .multiply_scalar(self.projection_scale, stream)?
            .reshape(&split, stream)?;
        let projected = self.projection_norm.apply(&projected, self.epsilon, stream)?;
        projected
            .add(&embedded, stream)?
            .multiply_scalar(std::f32::consts::FRAC_1_SQRT_2, stream)
    }
}

/// The layer's half: `h + norm(projection(gelu(gate(h)) * input))` after its
/// feed-forward, before the layer scalar.
#[derive(Debug)]
pub(in crate::engine::hybrid::moe) struct PerLayerGate {
    gate: BoundLinear,
    projection: BoundLinear,
    output_norm: NormWeight,
    index: usize,
    step: Arc<StepShare>,
}

impl PerLayerGate {
    pub(in crate::engine::hybrid::moe) fn load(
        tensors: &ModelTensors,
        bindings: PerLayerInputBindings<'_>,
        index: usize,
        step: Arc<StepShare>,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            gate: BoundLinear::load(tensors, bindings.gate, stream)?,
            projection: BoundLinear::load(tensors, bindings.projection, stream)?,
            output_norm: NormWeight::load_name(tensors, &bindings.output_norm.source)?,
            index,
            step,
        })
    }

    pub(in crate::engine::hybrid::moe) fn apply(
        &self,
        hidden: &Array,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Array> {
        let input = self.step.per_layer(self.index, stream)?;
        let gated = self.gate.forward(hidden, stream)?.gelu_approx_mul(&input, stream)?;
        let projected = self.projection.forward(&gated, stream)?;
        hidden.add(&self.output_norm.apply(&projected, epsilon, stream)?, stream)
    }
}
