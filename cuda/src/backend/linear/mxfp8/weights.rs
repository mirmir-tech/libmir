use models::weights::{HybridMoeExpertBindings, RoutedExpertBindings};

use super::MxFp8CheckpointWeight;
use crate::{CudaTensorSet, Error, Result, backend::tuning::MxFp8MoeStorage};

#[derive(Clone, Debug)]
pub(super) enum MxFp8GateUpWeights {
    Separate {
        gate: Box<MxFp8CheckpointWeight>,
        up: Box<MxFp8CheckpointWeight>,
    },
    Interleaved {
        gate_up: Box<MxFp8CheckpointWeight>,
    },
}

#[derive(Clone, Debug)]
/// Typed OCP MXFP8 matrix banks for a routed gated MLP.
pub struct MxFp8ExpertWeights {
    pub(super) gate_up: MxFp8GateUpWeights,
    pub(super) down: MxFp8CheckpointWeight,
    experts: usize,
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
}

impl MxFp8ExpertWeights {
    pub fn load_hybrid(
        tensors: &CudaTensorSet,
        bindings: &HybridMoeExpertBindings<'_>,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let HybridMoeExpertBindings::Stacked(weights) = bindings else {
            return Err(Error::UnsupportedDecoderLayer(
                "gathered MXFP8 hybrid experts require separate stacked banks".into(),
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
                MxFp8GateUpWeights::Separate {
                    gate: Box::new(MxFp8CheckpointWeight::load_binding(tensors, gate)?),
                    up: Box::new(MxFp8CheckpointWeight::load_binding(tensors, up)?),
                },
                down,
            ),
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => (
                MxFp8GateUpWeights::Interleaved {
                    gate_up: Box::new(MxFp8CheckpointWeight::load_binding(tensors, gate_up)?),
                },
                down,
            ),
            RoutedExpertBindings::Individual { .. } => {
                return Err(Error::UnsupportedDecoderLayer(
                    "MXFP8 CUDA experts require stacked checkpoint bindings".into(),
                ));
            },
        };
        let weights = Self {
            gate_up,
            down: MxFp8CheckpointWeight::load_binding(tensors, down)?,
            experts,
            hidden,
            intermediate,
        };
        weights.validate()?;
        Ok(weights)
    }

    fn validate(&self) -> Result<()> {
        match &self.gate_up {
            MxFp8GateUpWeights::Separate { gate, up } => {
                gate.validate_bank(self.experts, self.hidden, self.intermediate)?;
                up.validate_bank(self.experts, self.hidden, self.intermediate)?;
            },
            MxFp8GateUpWeights::Interleaved { gate_up } => {
                gate_up.validate_interleaved_bank(self.experts, self.hidden, self.intermediate)?;
            },
        }
        self.down.validate_bank(self.experts, self.intermediate, self.hidden)
    }

    pub(super) fn intermediate_elements(&self, assignments: usize) -> Result<usize> {
        assignments
            .checked_mul(self.intermediate)
            .ok_or(Error::InvalidDecoderKernel("MXFP8 expert intermediate size overflow"))
    }

    pub(super) fn routed_output_elements(&self, assignments: usize) -> Result<usize> {
        assignments
            .checked_mul(self.hidden)
            .ok_or(Error::InvalidDecoderKernel("MXFP8 routed output size overflow"))
    }

    pub(super) const fn geometry(&self) -> (usize, usize, usize) {
        (self.experts, self.hidden, self.intermediate)
    }

    pub(super) const fn storage(&self) -> MxFp8MoeStorage {
        match self.gate_up {
            MxFp8GateUpWeights::Separate { .. } => MxFp8MoeStorage::Separate,
            MxFp8GateUpWeights::Interleaved { .. } => MxFp8MoeStorage::Interleaved,
        }
    }

    pub(super) fn has_bias(&self) -> bool {
        let gate_up = match &self.gate_up {
            MxFp8GateUpWeights::Separate { gate, up } => gate.bias.is_some() || up.bias.is_some(),
            MxFp8GateUpWeights::Interleaved { gate_up } => gate_up.bias.is_some(),
        };
        gate_up || self.down.bias.is_some()
    }
}
