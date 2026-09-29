use mircup::{AttentionWindow, LayerNorm, Linear, Tensor, attention, gelu, relu};
use models::decision::{DecisionTensor, HeadTensor, ScorerTensor};

use super::{batch::PaddedBatch, weights::WeightReader};
use crate::Result;

/// The decision head: a question-kind embedding, pre-norm transformer layers
/// with `ReLU` feed-forward blocks, and an option scorer read at the markers.
pub struct Head {
    kinds: Tensor,
    layers: Vec<Layer>,
    scorer_norm: LayerNorm,
    scorer_hidden: Linear,
    scorer_logit: Linear,
    heads: usize,
}

struct Layer {
    attention_norm: LayerNorm,
    qkv: Linear,
    output: Linear,
    feed_forward_norm: LayerNorm,
    up: Linear,
    down: Linear,
}

impl Head {
    pub fn load(reader: &WeightReader<'_>, layers: usize, heads: usize) -> Result<Self> {
        let layers = (0..layers)
            .map(|layer| {
                let role = |tensor| DecisionTensor::Head { layer, tensor };
                let linear = |weight, bias| reader.linear(role(weight), Some(role(bias)));
                Ok(Layer {
                    attention_norm: reader.biased_norm(
                        role(HeadTensor::AttentionNormWeight),
                        role(HeadTensor::AttentionNormBias),
                    )?,
                    qkv: linear(HeadTensor::QkvWeight, HeadTensor::QkvBias)?,
                    output: linear(
                        HeadTensor::AttentionOutputWeight,
                        HeadTensor::AttentionOutputBias,
                    )?,
                    feed_forward_norm: reader.biased_norm(
                        role(HeadTensor::FeedForwardNormWeight),
                        role(HeadTensor::FeedForwardNormBias),
                    )?,
                    up: linear(HeadTensor::UpWeight, HeadTensor::UpBias)?,
                    down: linear(HeadTensor::DownWeight, HeadTensor::DownBias)?,
                })
            })
            .collect::<Result<_>>()?;
        let scorer = DecisionTensor::Scorer;
        Ok(Self {
            kinds: reader.tensor(DecisionTensor::TypeEmbedding)?,
            layers,
            scorer_norm: reader
                .biased_norm(scorer(ScorerTensor::NormWeight), scorer(ScorerTensor::NormBias))?,
            scorer_hidden: reader.linear(
                scorer(ScorerTensor::HiddenWeight),
                Some(scorer(ScorerTensor::HiddenBias)),
            )?,
            scorer_logit: reader
                .linear(scorer(ScorerTensor::LogitWeight), Some(scorer(ScorerTensor::LogitBias)))?,
            heads,
        })
    }

    /// Option logits of every row, one per marker, from encoder states
    /// `[rows × length, hidden]`.
    pub fn logits(&self, mut hidden: Tensor, batch: &PaddedBatch) -> Result<Vec<Vec<f32>>> {
        let width = hidden.width();
        for (row, states) in hidden.data_mut().chunks_exact_mut(batch.length * width).enumerate() {
            let kind = batch.kinds[row];
            let embedding = &self.kinds.data()[kind * width..(kind + 1) * width];
            for token in states.chunks_exact_mut(width) {
                token.iter_mut().zip(embedding).for_each(|(value, addend)| *value += addend);
            }
        }
        for layer in &self.layers {
            let normalized = layer.attention_norm.forward(&hidden)?;
            layer.output.accumulate(&self.attend(layer, &normalized, batch)?, &mut hidden)?;
            let mut expanded = layer.up.forward(&layer.feed_forward_norm.forward(&hidden)?)?;
            relu(&mut expanded);
            layer.down.accumulate(&expanded, &mut hidden)?;
        }
        let positions: Vec<usize> = batch
            .markers
            .iter()
            .enumerate()
            .flat_map(|(row, markers)| {
                markers.iter().map(move |marker| row * batch.length + marker)
            })
            .collect();
        let mut scored = self
            .scorer_hidden
            .forward(&self.scorer_norm.forward(&hidden.gather_rows(&positions)?)?)?;
        gelu(&mut scored);
        let mut logits = self.scorer_logit.forward(&scored)?.into_data().into_iter();
        Ok(batch
            .markers
            .iter()
            .map(|markers| logits.by_ref().take(markers.len()).collect())
            .collect())
    }

    fn attend(&self, layer: &Layer, input: &Tensor, batch: &PaddedBatch) -> Result<Tensor> {
        let rows = batch.lengths.len();
        let qkv = layer.qkv.forward(input)?.reshape(vec![rows, batch.length, 3 * input.width()])?;
        let mixed = attention(&qkv, self.heads, &batch.lengths, AttentionWindow::Full)?;
        Ok(mixed.reshape(vec![rows * batch.length, input.width()])?)
    }
}
