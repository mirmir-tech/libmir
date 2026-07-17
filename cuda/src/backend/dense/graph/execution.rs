use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::super::{DecodeDenseSwiGlu, DenseSwiGluWeights, GateUpBuffers};
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
        self.hidden_ops.add(
            &self.stream,
            input,
            &self.scratch.attention,
            &mut self.scratch.residual,
        )?;
        let separate = self.gate_up.execute(
            &self.scratch.residual,
            &self.post_attention_norm,
            weights.post_attention_norm,
            weights.gate_up,
            &mut GateUpBuffers {
                normalized: &mut self.scratch.normalized,
                packed: &mut self.scratch.gate_up,
                separate: &mut self.scratch.gate_up_separate,
            },
        )?;
        let fused_down = separate
            && self.down.execute_gated(
                &self.scratch.gate_up_separate[0],
                &self.scratch.gate_up_separate[1],
                self.config.activation,
                weights.down,
                &mut self.scratch.mlp,
            )?;
        if separate && !fused_down {
            self.activation.execute_separate(
                &self.stream,
                &self.scratch.gate_up_separate[0],
                &self.scratch.gate_up_separate[1],
                &mut self.scratch.activated,
                self.config.activation.into(),
            )?;
        } else if !fused_down {
            self.activation.execute(
                &self.stream,
                &self.scratch.gate_up,
                &mut self.scratch.activated,
                self.config.activation.into(),
            )?;
        }
        if !fused_down {
            self.down
                .execute(&self.scratch.activated, weights.down, &mut self.scratch.mlp)?;
        }
        self.hidden_ops
            .add(&self.stream, &self.scratch.residual, &self.scratch.mlp, output)?;
        Ok(nodes)
    }
}
