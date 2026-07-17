use super::{
    BlockTable, DecodeMoeBlockBf16, DecodeMoeBlockWeights, DeviceBuffer, KvWritePlan, Nodes,
    Result, bf16,
};

impl DecodeMoeBlockBf16 {
    pub(super) fn execute_captured(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<Nodes> {
        let nodes = self.attention.execute_captured(
            input,
            weights.attention,
            write_plan,
            table,
            &mut self.scratch.attention,
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
            super::super::scalar(weights.layer_scalar)?,
            output,
        )?;
        Ok(nodes)
    }
}
