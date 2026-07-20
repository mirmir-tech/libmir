use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::PrefillMoeBlockBf16;
use crate::{
    DecodeMoeBlockBf16, DecodeMoeBlockWeights, Result,
    backend::{attention::ImageAttentionSpan, block::scalar},
};

impl PrefillMoeBlockBf16 {
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        state: &mut DecodeMoeBlockBf16,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_masked(state, input, weights, write_plan, table, start_position, output, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_masked(
        &mut self,
        state: &mut DecodeMoeBlockBf16,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
        image: Option<ImageAttentionSpan>,
    ) -> Result<()> {
        self.attention.execute_masked(
            &mut state.attention,
            input,
            weights.attention,
            write_plan,
            table,
            start_position,
            &mut self.scratch.attention,
            image,
        )?;
        self.post_attention_norm.execute(
            &self.scratch.attention,
            weights.post_attention_norm,
            &mut self.scratch.attention_norm,
        )?;
        self.hidden_ops.add(
            &self.stream,
            input,
            &self.scratch.attention_norm,
            &mut self.scratch.hidden,
        )?;
        self.execute_dense(weights)?;
        self.execute_experts(weights)?;
        self.hidden_ops.add(
            &self.stream,
            &self.scratch.dense,
            &self.scratch.expert_norm,
            &mut self.scratch.feed_forward,
        )?;
        self.post_feed_forward_norm.execute(
            &self.scratch.feed_forward,
            weights.post_feed_forward_norm,
            &mut self.scratch.feed_forward_norm,
        )?;
        self.hidden_ops.add(
            &self.stream,
            &self.scratch.hidden,
            &self.scratch.feed_forward_norm,
            &mut self.scratch.residual,
        )?;
        self.hidden_ops.multiply_scalar(
            &self.stream,
            &self.scratch.residual,
            scalar(weights.layer_scalar)?,
            output,
        )
    }

    fn execute_dense(&mut self, weights: DecodeMoeBlockWeights<'_>) -> Result<()> {
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

    fn execute_experts(&mut self, weights: DecodeMoeBlockWeights<'_>) -> Result<()> {
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

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}
