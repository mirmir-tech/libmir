//! The always-active shared expert beside the routed ones (Qwen2-MoE,
//! GLM-4.5), and its decode-time gate/up fusion.

use super::{Array, FusedGateUp, Result, SharedExpertMoe, Stream};
use crate::{
    FusionMode,
    engine::{
        binding::BoundLinear, fusion_planner::FusionPlanner, gate_up_tuning,
        lowering::FeedForwardLowering,
    },
};

#[derive(Debug)]
pub(super) struct SharedExpert {
    pub(super) gate: BoundLinear,
    pub(super) up: BoundLinear,
    pub(super) fused_gate_up: Option<FusedGateUp>,
    pub(super) down: BoundLinear,
    /// Sigmoid gate on the shared expert's output; `None` adds it ungated.
    pub(super) output_gate: Option<BoundLinear>,
    pub(super) gate_up_mode: FusionMode,
}

impl SharedExpert {
    pub(super) fn new(
        [gate, up, down]: [BoundLinear; 3],
        output_gate: Option<BoundLinear>,
        lowering: FeedForwardLowering,
        stream: &Stream,
    ) -> Self {
        let planner = FusionPlanner::new(stream);
        let fusion = planner.projections(lowering, super::load::projection_biases(&gate, &up));
        Self {
            gate,
            up,
            fused_gate_up: None,
            down,
            output_gate,
            gate_up_mode: if fusion.gate_up {
                planner.shared_dense_gate_up_mode()
            } else {
                FusionMode::Disabled
            },
        }
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let fused = self.fused_gate_up.as_ref();
        let (gate, up) = if gate_up_tuning::is_single_token(input)? {
            match (self.gate_up_mode, fused) {
                (FusionMode::Auto, Some(fused)) => {
                    gate_up_tuning::forward_decode_plan(&self.gate, &self.up, fused, input, stream)?
                },
                _ => gate_up_tuning::forward(&self.gate, &self.up, fused, input, stream)?,
            }
        } else if self.gate_up_mode == FusionMode::Enabled {
            fused.map_or_else(
                || self.separate(input, stream),
                |fused| fused.forward_pair(input, stream),
            )?
        } else {
            self.separate(input, stream)?
        };
        let output = self.down.forward(&gate.silu_mul(&up, stream)?, stream)?;
        match &self.output_gate {
            Some(gate) => gate.forward(input, stream)?.sigmoid_mul(&output, stream),
            None => Ok(output),
        }
    }

    fn separate(&self, input: &Array, stream: &Stream) -> Result<(Array, Array)> {
        Ok((self.gate.forward(input, stream)?, self.up.forward(input, stream)?))
    }
}

impl SharedExpertMoe {
    pub(crate) fn decode_plan_candidate_bytes(&self) -> Result<Option<usize>> {
        self.shared
            .as_ref()
            .map_or(Ok(None), |shared| shared.gate.fused_gate_up_bytes(&shared.up))
    }

    pub(crate) fn enable_decode_plan_candidate(&mut self, stream: &Stream) -> Result<bool> {
        let Some(shared) = self.shared.as_mut() else {
            return Ok(false);
        };
        shared.fused_gate_up = shared.gate.fuse_gate_up(&shared.up, stream)?;
        shared.fused_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        Ok(shared.fused_gate_up.is_some())
    }

    pub(crate) fn has_decode_plan_candidates(&self) -> bool {
        self.shared.as_ref().is_some_and(|shared| {
            shared.gate_up_mode == FusionMode::Auto && shared.fused_gate_up.is_some()
        })
    }
}
