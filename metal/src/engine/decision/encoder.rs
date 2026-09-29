use models::{
    decision::{DecisionTensor, EncoderTensor},
    layout::{ModernBertAttention, ModernBertConfig},
};

use super::{
    attention::Heads,
    batch::PaddedBatch,
    weights::{Linear, Norm, WeightReader},
};
use crate::engine::{Array, Dtype, Error, Result, Stream};

/// A `ModernBERT` encoder evaluated on one padded batch.
pub struct Encoder {
    embeddings: Array,
    embedding_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    heads: Heads,
    intermediate: usize,
    global_base: f32,
    local_base: f32,
}

struct Layer {
    /// `None` on layer 0, which attends to the embedding norm output.
    attention_norm: Option<Norm>,
    qkv: Linear,
    output: Linear,
    mlp_norm: Norm,
    mlp_input: Linear,
    mlp_output: Linear,
    attention: ModernBertAttention,
}

impl Encoder {
    pub fn load(config: &ModernBertConfig, reader: &WeightReader<'_>) -> Result<Self> {
        let eps = config.norm_eps.to_string().parse()?;
        let layers = config
            .layers
            .iter()
            .enumerate()
            .map(|(layer, &attention)| {
                let role = |tensor| DecisionTensor::Encoder { layer, tensor };
                Ok(Layer {
                    attention_norm: (layer > 0)
                        .then(|| reader.unbiased_norm(role(EncoderTensor::AttentionNorm), eps))
                        .transpose()?,
                    qkv: reader.linear(role(EncoderTensor::Qkv), None)?,
                    output: reader.linear(role(EncoderTensor::AttentionOutput), None)?,
                    mlp_norm: reader.unbiased_norm(role(EncoderTensor::MlpNorm), eps)?,
                    mlp_input: reader.linear(role(EncoderTensor::MlpInput), None)?,
                    mlp_output: reader.linear(role(EncoderTensor::MlpOutput), None)?,
                    attention,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            embeddings: reader.stored(DecisionTensor::TokenEmbedding)?,
            embedding_norm: reader.unbiased_norm(DecisionTensor::EmbeddingNorm, eps)?,
            layers,
            final_norm: reader.unbiased_norm(DecisionTensor::FinalNorm, eps)?,
            heads: Heads {
                heads: i32::try_from(config.num_attention_heads)?,
                head_dim: i32::try_from(config.head_dim)?,
            },
            intermediate: config.intermediate_size,
            global_base: config.global_rope_theta.to_string().parse()?,
            local_base: config.local_rope_theta.to_string().parse()?,
        })
    }

    /// Final hidden states `[rows, length, hidden]`.
    pub fn forward(&self, batch: &PaddedBatch, stream: &Stream) -> Result<Array> {
        let embedded =
            self.embeddings.take(&batch.tokens, 0, stream)?.astype(Dtype::Float32, stream)?;
        let mut hidden = self.embedding_norm.forward(&embedded, stream)?;
        let bands = self.band_masks(batch)?;
        for layer in &self.layers {
            let (base, mask) = match layer.attention {
                ModernBertAttention::Global => (self.global_base, &batch.full),
                ModernBertAttention::Local { radius } => {
                    let band = bands.iter().find(|(band, _)| *band == radius);
                    (
                        self.local_base,
                        &band.ok_or(Error::InvalidModel("missing band mask".into()))?.1,
                    )
                },
            };
            let qkv = match &layer.attention_norm {
                Some(norm) => layer.qkv.forward(&norm.forward(&hidden, stream)?, stream)?,
                None => layer.qkv.forward(&hidden, stream)?,
            };
            let mixed = self.heads.attend(&qkv, batch, Some(base), mask, stream)?;
            hidden = hidden.add(&layer.output.forward(&mixed, stream)?, stream)?;
            let expanded =
                layer.mlp_input.forward(&layer.mlp_norm.forward(&hidden, stream)?, stream)?;
            hidden = hidden.add(
                &layer.mlp_output.forward(&self.geglu(&expanded, batch, stream)?, stream)?,
                stream,
            )?;
        }
        self.final_norm.forward(&hidden, stream)
    }

    fn band_masks(&self, batch: &PaddedBatch) -> Result<Vec<(usize, Array)>> {
        let mut radii: Vec<usize> = self
            .layers
            .iter()
            .filter_map(|layer| match layer.attention {
                ModernBertAttention::Local { radius } => Some(radius),
                ModernBertAttention::Global => None,
            })
            .collect();
        radii.sort_unstable();
        radii.dedup();
        radii.into_iter().map(|radius| Ok((radius, batch.band(radius)?))).collect()
    }

    fn geglu(&self, expanded: &Array, batch: &PaddedBatch, stream: &Stream) -> Result<Array> {
        let (rows, length) = (usize::try_from(batch.rows)?, usize::try_from(batch.length)?);
        let width = self.intermediate;
        let input = expanded.slice(&[0, 0, 0], &[rows, length, width], stream)?;
        let gate = expanded.slice(&[0, 0, width], &[rows, length, 2 * width], stream)?;
        input.gelu(stream)?.multiply(&gate, stream)
    }
}
