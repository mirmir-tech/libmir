use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::super::{DecodeDenseSwiGlu, DenseSwiGluWeights};
use crate::{Result, backend::attention::graph::Nodes};

impl DecodeDenseSwiGlu {
    pub(super) fn execute_captured(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
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
        self.feed_forward.execute(input, weights, &mut self.scratch, output)?;
        Ok(nodes)
    }
}
