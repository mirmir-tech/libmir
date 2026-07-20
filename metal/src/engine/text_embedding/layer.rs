use models::weights::TextTensorLayout;

use super::{LayerConfig, attention, weights::LayerWeights};
use crate::engine::{Array, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct TextEmbeddingLayer {
    config: LayerConfig,
    weights: LayerWeights,
}

impl TextEmbeddingLayer {
    pub fn load(
        tensors: &ModelTensors,
        config: LayerConfig,
        layout: &TextTensorLayout,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            config,
            weights: LayerWeights::load(tensors, config, layout, stream)?,
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention =
            attention::forward(&normalized, &self.weights.attention, self.config, stream)?;
        let hidden = input.add(&attention, stream)?;
        let normalized =
            self.weights
                .post_attention_norm
                .apply(&hidden, self.config.rms_norm_eps, stream)?;
        let gate = self.weights.gate.forward(&normalized, stream)?;
        let up = self.weights.up.forward(&normalized, stream)?;
        let activated = gate.silu_mul(&up, stream)?;
        hidden.add(&self.weights.down.forward(&activated, stream)?, stream)
    }
}
