//! The shared experts: gate/up, activation, down projection and the
//! optional sigmoid output gate.

use mircuda::{DeviceBuffer, bf16};

use super::{CudaAffineSharedExpertMoeExecution, SharedGateUp};
use crate::Result;

impl CudaAffineSharedExpertMoeExecution {
    pub(super) fn shared(&mut self, input: &DeviceBuffer<bf16>) -> Result<()> {
        let stream = self
            .parallel
            .as_ref()
            .map_or(&self.backend.inner.stream, |parallel| &parallel.stream);
        match &mut self.shared_gate_up {
            SharedGateUp::Separate { gate, up } => {
                gate.execute(input, &mut self.scratch.shared_gate)?;
                up.execute(input, &mut self.scratch.shared_up)?;
                self.shared_activation.gated(
                    stream,
                    &self.scratch.shared_gate,
                    &self.scratch.shared_up,
                    &mut self.scratch.shared_intermediate,
                    self.config.activation.into(),
                )?;
            },
            SharedGateUp::PackedNvFp4(operation) => {
                operation.execute(
                    input,
                    &mut self.scratch.shared_gate_up,
                    mircuda::MarlinNvFp4ThreadConfig::N128K64,
                )?;
                self.shared_activation.gated_concatenated(
                    stream,
                    &self.scratch.shared_gate_up,
                    &mut self.scratch.shared_intermediate,
                    self.config.shared_intermediate_size,
                    self.config.activation.into(),
                )?;
            },
        }
        let Some(output_gate) = &mut self.shared_output_gate else {
            return self
                .shared_down
                .execute(&self.scratch.shared_intermediate, &mut self.scratch.gated_shared_output);
        };
        self.shared_down
            .execute(&self.scratch.shared_intermediate, &mut self.scratch.shared_output)?;
        output_gate.execute(input, &mut self.scratch.shared_output_gate)?;
        self.sigmoid_multiply.execute(
            stream,
            &self.scratch.shared_output,
            &self.scratch.shared_output_gate,
            &mut self.scratch.gated_shared_output,
        )
    }
}
