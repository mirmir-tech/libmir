use mircup::{AttentionWindow, EmbeddingTable, LayerNorm, Linear, Rope, Tensor, attention, gelu};
use models::{
    decision::{DecisionTensor, EncoderTensor},
    layout::{ModernBertAttention, ModernBertConfig},
};

use super::{batch::PaddedBatch, weights::WeightReader};
use crate::Result;

/// A `ModernBERT` encoder: embedding norm, pre-norm layers with banded or
/// global attention and `GeGLU` feed-forward blocks, and a final norm.
pub struct Encoder {
    embeddings: EmbeddingTable,
    embedding_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    global_rope: Rope,
    local_rope: Rope,
    heads: usize,
}

struct Layer {
    /// `None` on layer 0, which attends to the embedding norm output.
    attention_norm: Option<LayerNorm>,
    qkv: Linear,
    output: Linear,
    mlp_norm: LayerNorm,
    mlp_input: Linear,
    mlp_output: Linear,
    attention: ModernBertAttention,
}

impl Encoder {
    pub fn load(
        config: &ModernBertConfig,
        reader: &WeightReader<'_>,
        positions: usize,
    ) -> Result<Self> {
        let layers = config
            .layers
            .iter()
            .enumerate()
            .map(|(layer, &attention)| {
                let role = |tensor| DecisionTensor::Encoder { layer, tensor };
                Ok(Layer {
                    attention_norm: (layer > 0)
                        .then(|| reader.encoder_norm(role(EncoderTensor::AttentionNorm)))
                        .transpose()?,
                    qkv: reader.linear(role(EncoderTensor::Qkv), None)?,
                    output: reader.linear(role(EncoderTensor::AttentionOutput), None)?,
                    mlp_norm: reader.encoder_norm(role(EncoderTensor::MlpNorm))?,
                    mlp_input: reader.linear(role(EncoderTensor::MlpInput), None)?,
                    mlp_output: reader.linear(role(EncoderTensor::MlpOutput), None)?,
                    attention,
                })
            })
            .collect::<Result<_>>()?;
        let rope = |theta| Rope::new(config.head_dim, rope_base(theta), positions);
        Ok(Self {
            embeddings: reader.embedding(DecisionTensor::TokenEmbedding)?,
            embedding_norm: reader.encoder_norm(DecisionTensor::EmbeddingNorm)?,
            layers,
            final_norm: reader.encoder_norm(DecisionTensor::FinalNorm)?,
            global_rope: rope(config.global_rope_theta)?,
            local_rope: rope(config.local_rope_theta)?,
            heads: config.num_attention_heads,
        })
    }

    /// Final hidden states `[rows × length, hidden]` of a padded batch.
    pub fn forward(&self, batch: &PaddedBatch) -> Result<Tensor> {
        let mut hidden = self.embeddings.lookup(&batch.tokens)?;
        self.embedding_norm.forward_in_place(&mut hidden)?;
        for layer in &self.layers {
            let input = match &layer.attention_norm {
                Some(norm) => norm.forward(&hidden)?,
                None => hidden.clone(),
            };
            hidden.add_assign(&self.attend(layer, &input, batch)?)?;
            let normalized = layer.mlp_norm.forward(&hidden)?;
            hidden.add_assign(&feed_forward(layer, &normalized)?)?;
        }
        self.final_norm.forward_in_place(&mut hidden)?;
        Ok(hidden)
    }

    fn attend(&self, layer: &Layer, input: &Tensor, batch: &PaddedBatch) -> Result<Tensor> {
        let qkv = layer.qkv.forward(input)?;
        let hidden = input.width();
        let shape = vec![batch.lengths.len(), batch.length, self.heads, hidden / self.heads];
        let [query, key, value] = qkv.split_last::<3>()?;
        let mut query = query.reshape(shape.clone())?;
        let mut key = key.reshape(shape.clone())?;
        let value = value.reshape(shape)?;
        let (rope, window) = match layer.attention {
            ModernBertAttention::Global => (&self.global_rope, AttentionWindow::Full),
            ModernBertAttention::Local { radius } => {
                (&self.local_rope, AttentionWindow::Band { radius })
            },
        };
        rope.apply(&mut query)?;
        rope.apply(&mut key)?;
        let mixed = attention(&query, &key, &value, &batch.lengths, window)?;
        Ok(layer
            .output
            .forward(&mixed.reshape(vec![batch.lengths.len() * batch.length, hidden])?)?)
    }
}

fn feed_forward(layer: &Layer, input: &Tensor) -> Result<Tensor> {
    let [mut activated, gate] = layer.mlp_input.forward(input)?.split_last::<2>()?;
    gelu(&mut activated);
    activated.mul_assign(&gate)?;
    Ok(layer.mlp_output.forward(&activated)?)
}

/// Rotary bases are `f32` in the reference runtime's frequency tables.
#[expect(clippy::cast_possible_truncation, reason = "rotary bases are computed in f32")]
const fn rope_base(theta: f64) -> f32 {
    theta as f32
}
