use super::BatchedDecodeMoeBlockBf16;
use crate::{DecodeMoeBlockWeights, Result};

impl BatchedDecodeMoeBlockBf16 {
    pub(super) fn execute_dense(&mut self, weights: DecodeMoeBlockWeights<'_>) -> Result<()> {
        self.pre_dense_norm.execute(
            &self.scratch.hidden,
            weights.pre_dense_norm,
            &mut self.scratch.normalized,
        )?;
        self.dense_gate_up.execute(
            &self.scratch.normalized,
            weights.dense_gate_up,
            &mut self.scratch.dense_gate_up,
        )?;
        self.dense_activation.execute(
            &self.stream,
            &self.scratch.dense_gate_up,
            &mut self.scratch.dense_activated,
            self.config.activation.into(),
        )?;
        self.dense_down.execute(
            &self.scratch.dense_activated,
            weights.dense_down,
            &mut self.scratch.normalized,
        )?;
        self.post_dense_norm.execute(
            &self.scratch.normalized,
            weights.post_dense_norm,
            &mut self.scratch.dense,
        )
    }

    pub(super) fn execute_experts(&mut self, weights: DecodeMoeBlockWeights<'_>) -> Result<()> {
        let selection = self.router.execute(&self.scratch.hidden, weights.router)?;
        self.pre_expert_norm.execute(
            &self.scratch.hidden,
            weights.pre_expert_norm,
            &mut self.scratch.normalized,
        )?;
        self.experts.execute(
            &self.scratch.normalized,
            selection.indices,
            selection.weights,
            &mut self.scratch.expert,
        )?;
        self.post_expert_norm.execute(
            &self.scratch.expert,
            weights.post_expert_norm,
            &mut self.scratch.expert_norm,
        )
    }
}
