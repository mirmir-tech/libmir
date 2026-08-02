use models::weights::{HybridMoeExpertBindings, RoutedExpertBindings};

use super::MxFp4CheckpointWeight;
use crate::{CudaTensorSet, Error, Result, backend::tuning::MxFp4MoeStorage};

#[derive(Clone, Debug)]
pub(super) enum MxFp4GateUpWeights {
    Separate {
        gate: MxFp4CheckpointWeight,
        up: MxFp4CheckpointWeight,
    },
    Interleaved {
        gate_up: MxFp4CheckpointWeight,
    },
}

#[derive(Clone, Debug)]
/// Typed OCP MXFP4 matrix banks for a routed gated MLP.
pub struct MxFp4ExpertWeights {
    pub(super) gate_up: MxFp4GateUpWeights,
    pub(super) down: MxFp4CheckpointWeight,
    experts: usize,
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
}

impl MxFp4ExpertWeights {
    pub fn load_hybrid(
        tensors: &CudaTensorSet,
        bindings: &HybridMoeExpertBindings<'_>,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let HybridMoeExpertBindings::Stacked(weights) = bindings else {
            return Err(Error::UnsupportedDecoderLayer(
                "gathered MXFP4 hybrid experts require separate stacked banks".into(),
            ));
        };
        Self::load(
            tensors,
            RoutedExpertBindings::SeparateGateUp {
                gate: weights.gate,
                up: weights.up,
                down: weights.down,
            },
            experts,
            hidden,
            intermediate,
        )
    }

    pub fn load(
        tensors: &CudaTensorSet,
        bindings: RoutedExpertBindings<'_>,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let (gate_up, down) = match bindings {
            RoutedExpertBindings::SeparateGateUp { gate, up, down } => (
                MxFp4GateUpWeights::Separate {
                    gate: MxFp4CheckpointWeight::load_binding(tensors, gate)?,
                    up: MxFp4CheckpointWeight::load_binding(tensors, up)?,
                },
                down,
            ),
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => (
                MxFp4GateUpWeights::Interleaved {
                    gate_up: MxFp4CheckpointWeight::load_binding(tensors, gate_up)?,
                },
                down,
            ),
            RoutedExpertBindings::Individual { .. } => {
                return Err(Error::UnsupportedDecoderLayer(
                    "MXFP4 CUDA experts require stacked checkpoint bindings".into(),
                ));
            },
        };
        let weights = Self {
            gate_up,
            down: MxFp4CheckpointWeight::load_binding(tensors, down)?,
            experts,
            hidden,
            intermediate,
        };
        weights.validate()?;
        Ok(weights)
    }

    fn validate(&self) -> Result<()> {
        match &self.gate_up {
            MxFp4GateUpWeights::Separate { gate, up } => {
                gate.validate_bank(self.experts, self.hidden, self.intermediate)?;
                up.validate_bank(self.experts, self.hidden, self.intermediate)?;
            },
            MxFp4GateUpWeights::Interleaved { gate_up } => {
                gate_up.validate_interleaved_bank(self.experts, self.hidden, self.intermediate)?;
            },
        }
        self.down.validate_bank(self.experts, self.intermediate, self.hidden)
    }

    pub(super) fn intermediate_elements(&self, assignments: usize) -> Result<usize> {
        assignments
            .checked_mul(self.intermediate)
            .ok_or(Error::InvalidDecoderKernel("MXFP4 expert intermediate size overflow"))
    }

    pub(super) fn routed_output_elements(&self, assignments: usize) -> Result<usize> {
        assignments
            .checked_mul(self.hidden)
            .ok_or(Error::InvalidDecoderKernel("MXFP4 routed output size overflow"))
    }

    pub(super) const fn geometry(&self) -> (usize, usize, usize) {
        (self.experts, self.hidden, self.intermediate)
    }

    pub(super) const fn storage(&self) -> MxFp4MoeStorage {
        match self.gate_up {
            MxFp4GateUpWeights::Separate { .. } => MxFp4MoeStorage::Separate,
            MxFp4GateUpWeights::Interleaved { .. } => MxFp4MoeStorage::Interleaved,
        }
    }
}
