use models::{layout::DecoderConfig, weights::TextTensorLayout};

use super::{LayerConfig, layer::TextEmbeddingLayer};
use crate::engine::{Array, DenseEmbedding, ModelTensors, NormWeight, Result, Stream};

#[derive(Debug)]
pub struct TextEmbeddingModel {
    embedding: DenseEmbedding,
    layers: Vec<TextEmbeddingLayer>,
    final_norm: NormWeight,
    hidden_size: usize,
    rms_norm_eps: f32,
}

impl TextEmbeddingModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        layout: &TextTensorLayout,
        stream: &Stream,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for index in 0..decoder.num_hidden_layers {
            layers.push(TextEmbeddingLayer::load(
                tensors,
                LayerConfig::from_decoder(index, decoder)?,
                layout,
                stream,
            )?);
        }
        Ok(Self {
            embedding: DenseEmbedding::load(tensors, &layout.name("embed_tokens"))?,
            layers,
            final_norm: NormWeight::load(tensors, &layout.name("norm"))?,
            hidden_size: decoder.hidden_size,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
        })
    }

    pub fn embed(&self, token_ids: &[u32], dimensions: usize, stream: &Stream) -> Result<Vec<f32>> {
        let sequence = token_ids.len();
        if sequence == 0 || dimensions == 0 || dimensions > self.hidden_size {
            return Err(crate::engine::Error::InvalidModel(
                "invalid text embedding input or dimensions".into(),
            ));
        }
        let mut hidden = self
            .embedding
            .lookup(&Array::from_u32(token_ids, &[1, i32::try_from(sequence)?])?, stream)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, stream)?;
        }
        let hidden = self.final_norm.apply(&hidden, self.rms_norm_eps, stream)?;
        let pooled = hidden
            .slice(&[0, sequence - 1, 0], &[1, sequence, dimensions], stream)?
            .reshape(&[1, i32::try_from(dimensions)?], stream)?
            .l2_normalize(-1, 1.0e-12, stream)?;
        pooled.to_vec_f32_on_stream(stream)
    }
}
