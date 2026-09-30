use mircuda::{DeviceBuffer, PinnedBuffer};
use models::decision::{DecisionTensor, HeadTensor, ScorerTensor};

use super::{
    batch::DeviceBatch,
    encoder::Device,
    weights::{DeviceLinear, DeviceNorm, Uploader, Write},
};
use crate::{
    Result,
    kernels::{DecisionAttentionInput, DecisionWindow},
};

/// The decision head: kind embedding, pre-norm transformer layers with
/// `ReLU` feed-forward blocks, and the option scorer.
pub struct Head {
    kinds: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    scorer_norm: DeviceNorm,
    scorer_hidden: DeviceLinear,
    scorer_logit: DeviceLinear,
    heads: usize,
    hidden: usize,
}

struct Layer {
    attention_norm: DeviceNorm,
    qkv: DeviceLinear,
    output: DeviceLinear,
    feed_forward_norm: DeviceNorm,
    up: DeviceLinear,
    down: DeviceLinear,
}

impl Head {
    pub fn load(
        uploader: &Uploader<'_>,
        layers: usize,
        heads: usize,
        hidden: usize,
    ) -> Result<Self> {
        let layers = (0..layers)
            .map(|layer| {
                let role = |tensor| DecisionTensor::Head { layer, tensor };
                let linear = |weight, bias| uploader.linear(role(weight), Some(role(bias)));
                let norm = |weight, bias| uploader.norm(role(weight), Some(role(bias)), 1e-5);
                Ok(Layer {
                    attention_norm: norm(
                        HeadTensor::AttentionNormWeight,
                        HeadTensor::AttentionNormBias,
                    )?,
                    qkv: linear(HeadTensor::QkvWeight, HeadTensor::QkvBias)?,
                    output: linear(
                        HeadTensor::AttentionOutputWeight,
                        HeadTensor::AttentionOutputBias,
                    )?,
                    feed_forward_norm: norm(
                        HeadTensor::FeedForwardNormWeight,
                        HeadTensor::FeedForwardNormBias,
                    )?,
                    up: linear(HeadTensor::UpWeight, HeadTensor::UpBias)?,
                    down: linear(HeadTensor::DownWeight, HeadTensor::DownBias)?,
                })
            })
            .collect::<Result<_>>()?;
        let scorer = DecisionTensor::Scorer;
        Ok(Self {
            kinds: uploader.float(DecisionTensor::TypeEmbedding)?,
            layers,
            scorer_norm: uploader.norm(
                scorer(ScorerTensor::NormWeight),
                Some(scorer(ScorerTensor::NormBias)),
                1e-5,
            )?,
            scorer_hidden: uploader.linear(
                scorer(ScorerTensor::HiddenWeight),
                Some(scorer(ScorerTensor::HiddenBias)),
            )?,
            scorer_logit: uploader
                .linear(scorer(ScorerTensor::LogitWeight), Some(scorer(ScorerTensor::LogitBias)))?,
            heads,
            hidden,
        })
    }

    /// Option logits per row. The last layer runs only at the markers, the
    /// only rows the scorer reads; its keys still cover every token.
    pub fn logits(
        &self,
        device: &Device<'_>,
        mut hidden: DeviceBuffer<f32>,
        batch: &DeviceBatch,
    ) -> Result<Vec<Vec<f32>>> {
        let stream = &device.backend.inner.stream;
        device.elementwise.add_kind(
            stream,
            &mut hidden,
            (&batch.kinds, &self.kinds, batch.length),
        )?;
        let (last, earlier) = match self.layers.split_last() {
            Some((last, earlier)) => (Some(last), earlier),
            None => (None, &[][..]),
        };
        for layer in earlier {
            let mixed = self.attend_all(device, layer, &hidden, batch)?;
            layer.output.apply(device.kit(), &mixed, &mut hidden, Write::Accumulate)?;
            feed_forward(device, layer, &mut hidden)?;
        }
        let mut markers = device.buffer(batch.markers.len() * self.hidden)?;
        device.elementwise.gather(stream, &hidden, &batch.markers, &mut markers)?;
        if let Some(layer) = last {
            let mixed = self.attend(device, layer, &hidden, batch, &batch.markers)?;
            layer.output.apply(device.kit(), &mixed, &mut markers, Write::Accumulate)?;
            feed_forward(device, layer, &mut markers)?;
        }
        let mut scored =
            device.linear(&self.scorer_hidden, &device.norm(&markers, &self.scorer_norm)?)?;
        device.elementwise.gelu(stream, &mut scored)?;
        let logits = device.linear(&self.scorer_logit, &scored)?;
        let mut host: PinnedBuffer<f32> =
            device.backend.inner.context.allocate_pinned(logits.len())?;
        stream.copy_to_host(&logits, &mut host)?;
        let mut values = host.to_vec()?.into_iter();
        Ok(batch
            .marker_counts
            .iter()
            .map(|&count| values.by_ref().take(count).collect())
            .collect())
    }

    fn attend_all(
        &self,
        device: &Device<'_>,
        layer: &Layer,
        hidden: &DeviceBuffer<f32>,
        batch: &DeviceBatch,
    ) -> Result<DeviceBuffer<f32>> {
        let qkv = device.linear(&layer.qkv, &device.norm(hidden, &layer.attention_norm)?)?;
        let mut mixed = device.buffer(batch.tokens() * self.hidden)?;
        let geometry = (batch.length, self.heads, self.hidden / self.heads, DecisionWindow::Full);
        device.dense.execute(
            &device.backend.inner.stream,
            (&qkv, &batch.lengths),
            geometry,
            &mut mixed,
        )?;
        Ok(mixed)
    }

    fn attend(
        &self,
        device: &Device<'_>,
        layer: &Layer,
        hidden: &DeviceBuffer<f32>,
        batch: &DeviceBatch,
        queries: &DeviceBuffer<u32>,
    ) -> Result<DeviceBuffer<f32>> {
        let qkv = device.linear(&layer.qkv, &device.norm(hidden, &layer.attention_norm)?)?;
        let mut mixed = device.buffer(queries.len() * self.hidden)?;
        let input = DecisionAttentionInput {
            qkv: &qkv,
            queries,
            lengths: &batch.lengths,
            length: batch.length,
            heads: self.heads,
            head_dim: self.hidden / self.heads,
            window: DecisionWindow::Full,
        };
        device.attention.execute(&device.backend.inner.stream, &input, &mut mixed)?;
        Ok(mixed)
    }
}

fn feed_forward(device: &Device<'_>, layer: &Layer, hidden: &mut DeviceBuffer<f32>) -> Result<()> {
    let mut expanded = device.linear(&layer.up, &device.norm(hidden, &layer.feed_forward_norm)?)?;
    device.elementwise.relu(&device.backend.inner.stream, &mut expanded)?;
    layer.down.apply(device.kit(), &expanded, hidden, Write::Accumulate)
}
