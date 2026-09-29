mod attention;
mod batch;
mod encoder;
mod head;
mod weights;

use models::decision::{DecisionCheckpoint, DecisionRow};

use crate::engine::{Result, Stream};

/// A Laya decision checkpoint resident on the GPU, evaluated in `f32`.
pub struct MetalDecisionModel {
    stream: Stream,
    encoder: encoder::Encoder,
    head: head::Head,
    positions: usize,
}

impl MetalDecisionModel {
    pub fn load(checkpoint: &DecisionCheckpoint) -> Result<Self> {
        let stream = Stream::new_gpu()?;
        let (encoder, head) = {
            let reader = weights::WeightReader::open(&checkpoint.tensors, &stream)?;
            (
                encoder::Encoder::load(&checkpoint.encoder, &reader)?,
                head::Head::load(
                    &reader,
                    checkpoint.agent.head_layers,
                    checkpoint.head_attention_heads(),
                    checkpoint.encoder.hidden_size,
                )?,
            )
        };
        Ok(Self {
            stream,
            encoder,
            head,
            positions: checkpoint.agent.max_len,
        })
    }

    /// Raw option logits of every row, one per marker.
    pub fn logits(&self, rows: &[DecisionRow]) -> Result<Vec<Vec<f32>>> {
        let batch = batch::PaddedBatch::new(rows, self.positions)?;
        let hidden = self.encoder.forward(&batch, &self.stream)?;
        self.head.logits(&hidden, &batch, &self.stream)
    }
}
