//! Feed-forward of a linear-attention layer: routed experts or a dense
//! `SwiGLU`.

use models::weights::DenseFeedForwardBindings;

#[cfg(test)]
use super::super::super::Error;
use super::super::super::{
    Array, ModelTensors, Result, SharedExpertMoe, Stream, binding::BoundLinear,
};

#[derive(Debug)]
pub(super) struct DenseMlp {
    gate: BoundLinear,
    up: BoundLinear,
    down: BoundLinear,
}

impl DenseMlp {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: DenseFeedForwardBindings<'_>,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            gate: BoundLinear::load(tensors, bindings.gate, stream)?,
            up: BoundLinear::load(tensors, bindings.up, stream)?,
            down: BoundLinear::load(tensors, bindings.down, stream)?,
        })
    }

    fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let gate = self.gate.forward(input, stream)?;
        let up = self.up.forward(input, stream)?;
        self.down.forward(&gate.silu_mul(&up, stream)?, stream)
    }
}

#[derive(Debug)]
pub(super) enum FeedForward {
    Routed(Box<SharedExpertMoe>),
    Dense(Box<DenseMlp>),
}

impl FeedForward {
    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        match self {
            Self::Routed(routed) => routed.forward(input, stream),
            Self::Dense(dense) => dense.forward(input, stream),
        }
    }

    pub(super) fn has_fused_routed_gate_up(&self) -> bool {
        match self {
            Self::Routed(routed) => routed.has_fused_routed_gate_up(),
            Self::Dense(_) => false,
        }
    }

    pub(super) fn has_decode_plan_candidates(&self) -> bool {
        match self {
            Self::Routed(routed) => routed.has_decode_plan_candidates(),
            Self::Dense(_) => false,
        }
    }

    pub(super) fn decode_plan_candidate_bytes(&self) -> Result<Option<usize>> {
        match self {
            Self::Routed(routed) => routed.decode_plan_candidate_bytes(),
            Self::Dense(_) => Ok(None),
        }
    }

    pub(super) fn enable_decode_plan_candidate(&mut self, stream: &Stream) -> Result<bool> {
        match self {
            Self::Routed(routed) => routed.enable_decode_plan_candidate(stream),
            Self::Dense(_) => Ok(false),
        }
    }

    pub(super) fn enable_routed_gate_up(&mut self, stream: &Stream) -> Result<bool> {
        match self {
            Self::Routed(routed) => routed.enable_routed_gate_up(stream),
            Self::Dense(_) => Ok(false),
        }
    }

    pub(super) fn fused_routed_gate_up_bytes(&self) -> Result<Option<usize>> {
        match self {
            Self::Routed(routed) => routed.fused_routed_gate_up_bytes(),
            Self::Dense(_) => Ok(None),
        }
    }

    /// Expert-only diagnostics; dense layers have no routing to probe.
    #[cfg(test)]
    pub(super) fn routed(&self) -> Result<&SharedExpertMoe> {
        match self {
            Self::Routed(routed) => Ok(routed.as_ref()),
            Self::Dense(_) => Err(Error::InvalidModel("dense layer has no routed experts".into())),
        }
    }
}
