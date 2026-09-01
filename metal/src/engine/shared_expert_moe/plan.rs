use super::{Array, Result, SharedExpertMoe, Stream};
use crate::{FusionMode, engine::gate_up_tuning};

impl SharedExpertMoe {
    pub(crate) fn decode_plan_candidate_bytes(&self) -> Result<Option<usize>> {
        self.shared_gate.fused_gate_up_bytes(&self.shared_up)
    }

    pub(crate) fn enable_decode_plan_candidate(&mut self, stream: &Stream) -> Result<bool> {
        self.fused_shared_gate_up = self.shared_gate.fuse_gate_up(&self.shared_up, stream)?;
        self.fused_shared_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        Ok(self.fused_shared_gate_up.is_some())
    }

    pub(super) fn shared(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let fused = self.fused_shared_gate_up.as_ref();
        let (gate, up) = if gate_up_tuning::is_single_token(input)? {
            match (self.shared_gate_up_mode, fused) {
                (FusionMode::Auto, Some(fused)) => gate_up_tuning::forward_decode_plan(
                    &self.shared_gate,
                    &self.shared_up,
                    fused,
                    input,
                    stream,
                )?,
                _ => gate_up_tuning::forward(
                    &self.shared_gate,
                    &self.shared_up,
                    fused,
                    input,
                    stream,
                )?,
            }
        } else if self.shared_gate_up_mode == FusionMode::Enabled {
            fused.map_or_else(
                || self.separate_shared(input, stream),
                |fused| fused.forward_pair(input, stream),
            )?
        } else {
            self.separate_shared(input, stream)?
        };
        let output = self.shared_down.forward(&gate.silu_mul(&up, stream)?, stream)?;
        self.shared_output_gate.forward(input, stream)?.sigmoid_mul(&output, stream)
    }

    pub(crate) fn has_decode_plan_candidates(&self) -> bool {
        self.shared_gate_up_mode == FusionMode::Auto && self.fused_shared_gate_up.is_some()
    }

    fn separate_shared(&self, input: &Array, stream: &Stream) -> Result<(Array, Array)> {
        Ok((self.shared_gate.forward(input, stream)?, self.shared_up.forward(input, stream)?))
    }
}
