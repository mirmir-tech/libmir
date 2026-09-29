mod batch;
mod encoder;
mod head;
mod weights;

use models::decision::{DecisionCheckpoint, DecisionRow};

use crate::Result;

/// A Laya decision checkpoint resident in host memory.
pub struct CpuDecisionModel {
    encoder: encoder::Encoder,
    head: head::Head,
    positions: usize,
}

impl CpuDecisionModel {
    pub fn load(checkpoint: &DecisionCheckpoint) -> Result<Self> {
        let eps = norm_eps(checkpoint.encoder.norm_eps);
        let reader = weights::WeightReader::new(&checkpoint.tensors, eps);
        let positions = checkpoint.agent.max_len;
        Ok(Self {
            encoder: encoder::Encoder::load(&checkpoint.encoder, &reader, positions)?,
            head: head::Head::load(
                &reader,
                checkpoint.agent.head_layers,
                checkpoint.head_attention_heads(),
            )?,
            positions,
        })
    }

    /// Raw option logits of every row, one per marker.
    pub fn logits(&self, rows: &[DecisionRow]) -> Result<Vec<Vec<f32>>> {
        let batch = batch::PaddedBatch::new(rows, self.positions)?;
        let hidden = self.encoder.forward(&batch)?;
        self.head.logits(hidden, &batch)
    }
}

#[expect(clippy::cast_possible_truncation, reason = "norm epsilons are applied in f32")]
const fn norm_eps(eps: f64) -> f32 {
    eps as f32
}
