use models::decision::{DecisionTensor, HeadTensor, ScorerTensor};

use super::{
    attention::Heads,
    batch::PaddedBatch,
    weights::{Linear, Norm, WeightReader},
};
use crate::engine::{Array, Result, Stream};

/// The decision head: a question-kind embedding, pre-norm transformer layers
/// with `ReLU` feed-forward blocks, and an option scorer read at the markers.
pub struct Head {
    kinds: Array,
    layers: Vec<Layer>,
    scorer_norm: Norm,
    scorer_hidden: Linear,
    scorer_logit: Linear,
    heads: Heads,
    zero: Array,
}

struct Layer {
    attention_norm: Norm,
    qkv: Linear,
    output: Linear,
    feed_forward_norm: Norm,
    up: Linear,
    down: Linear,
}

impl Head {
    pub fn load(
        reader: &WeightReader<'_>,
        layers: usize,
        heads: usize,
        hidden: usize,
    ) -> Result<Self> {
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
            kinds: reader.float(DecisionTensor::TypeEmbedding)?,
            layers,
            scorer_norm: reader
                .biased_norm(scorer(ScorerTensor::NormWeight), scorer(ScorerTensor::NormBias))?,
            scorer_hidden: reader.linear(
                scorer(ScorerTensor::HiddenWeight),
                Some(scorer(ScorerTensor::HiddenBias)),
            )?,
            scorer_logit: reader
                .linear(scorer(ScorerTensor::LogitWeight), Some(scorer(ScorerTensor::LogitBias)))?,
            heads: Heads {
                heads: i32::try_from(heads)?,
                head_dim: i32::try_from(hidden / heads)?,
            },
            zero: Array::from_f32(&[0.0], &[1])?,
        })
    }

    /// Option logits of every row, one per marker, from encoder states
    /// `[rows, length, hidden]`.
    pub fn logits(
        &self,
        hidden: &Array,
        batch: &PaddedBatch,
        stream: &Stream,
    ) -> Result<Vec<Vec<f32>>> {
        let width = self.heads.heads * self.heads.head_dim;
        let kinds = self
            .kinds
            .take(&batch.kinds, 0, stream)?
            .reshape(&[batch.rows, 1, width], stream)?;
        let mut hidden = hidden.add(&kinds, stream)?;
        for layer in &self.layers {
            let qkv = layer.qkv.forward(&layer.attention_norm.forward(&hidden, stream)?, stream)?;
            let mixed = self.heads.attend(&qkv, batch, None, &batch.full, stream)?;
            hidden = hidden.add(&layer.output.forward(&mixed, stream)?, stream)?;
            let expanded =
                layer.up.forward(&layer.feed_forward_norm.forward(&hidden, stream)?, stream)?;
            hidden =
                hidden.add(&layer.down.forward(&self.relu(&expanded, stream)?, stream)?, stream)?;
        }
        let markers = hidden
            .reshape(&[batch.rows * batch.length, width], stream)?
            .take(&batch.markers, 0, stream)?;
        let scored = self
            .scorer_hidden
            .forward(&self.scorer_norm.forward(&markers, stream)?, stream)?
            .gelu(stream)?;
        let mut logits =
            self.scorer_logit.forward(&scored, stream)?.to_vec_f32(stream)?.into_iter();
        Ok(batch
            .marker_counts
            .iter()
            .map(|&count| logits.by_ref().take(count).collect())
            .collect())
    }

    fn relu(&self, input: &Array, stream: &Stream) -> Result<Array> {
        Array::from_native(stream.native().graph().maximum(input.native(), self.zero.native())?)
    }
}
