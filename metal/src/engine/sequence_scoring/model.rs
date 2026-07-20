use models::layout::EncoderConfig;

use super::layer::EncoderLayer;
use crate::engine::{Array, DenseEmbedding, DenseLinear, LayerNorm, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct SequenceScoringModel {
    words: DenseEmbedding,
    token_types: Option<DenseEmbedding>,
    embedding_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    pooler: DenseLinear,
    classifier: DenseLinear,
}

impl SequenceScoringModel {
    pub fn load(tensors: &ModelTensors, config: &EncoderConfig, stream: &Stream) -> Result<Self> {
        if !config.packed_qkv || config.hidden_activation != "gelu" || config.num_labels != 1 {
            return Err(crate::engine::Error::InvalidModel(
                "unsupported sequence-scoring encoder contract".into(),
            ));
        }
        let eps = config.layer_norm_eps.to_string().parse()?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::load(tensors, index, config, stream)?);
        }
        Ok(Self {
            words: DenseEmbedding::load(tensors, "new.embeddings.word_embeddings")?,
            token_types: (config.type_vocab_size > 0)
                .then(|| DenseEmbedding::load(tensors, "new.embeddings.token_type_embeddings"))
                .transpose()?,
            embedding_norm: LayerNorm::load(tensors, "new.embeddings.LayerNorm", eps)?,
            layers,
            pooler: DenseLinear::load(tensors, "new.pooler.dense", stream)?,
            classifier: DenseLinear::load(tensors, "classifier", stream)?,
        })
    }

    pub fn score(&self, token_ids: &[u32], stream: &Stream) -> Result<f32> {
        if token_ids.is_empty() {
            return Err(crate::engine::Error::InvalidModel(
                "sequence scoring input is empty".into(),
            ));
        }
        let sequence = i32::try_from(token_ids.len())?;
        let ids = Array::from_u32(token_ids, &[1, sequence])?;
        let mut hidden = self.words.lookup(&ids, stream)?;
        if let Some(token_types) = &self.token_types {
            let zeros = vec![0; token_ids.len()];
            hidden = hidden.add(
                &token_types.lookup(&Array::from_u32(&zeros, &[1, sequence])?, stream)?,
                stream,
            )?;
        }
        hidden = self.embedding_norm.forward(&hidden, stream)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, stream)?;
        }
        let cls =
            hidden.slice(&[0, 0, 0], &[1, 1, usize::try_from(hidden.shape()?[2])?], stream)?;
        let pooled = self.pooler.forward(&cls, stream)?.tanh(stream)?;
        let score = self.classifier.forward(&pooled, stream)?.to_vec_f32_on_stream(stream)?;
        score.first().copied().ok_or_else(|| {
            crate::engine::Error::InvalidModel("sequence classifier returned no score".into())
        })
    }
}
