//! GGML expert banks read in place from GGUF files.

use mircuda::DeviceBuffer;
use models::weights::{
    BindingTransform, HybridMoeExpertBindings, RoutedExpertBindings, TensorBinding, TensorStorage,
};

use super::super::{super::RoutedExpertsConfig, ggml::GgmlExpertGeometry};
use crate::{
    CudaTensorSet, GgmlCheckpointWeight, Result,
    kernels::{GgmlBankRows, GgmlKernelType},
};

/// Stacked GGML banks `[experts, out, in]`, each bound as one matrix of
/// `experts * out` rows.
#[derive(Clone, Debug)]
pub(in crate::backend) struct GgmlExpertWeights {
    gate_up: GgmlGateUp,
    down: GgmlCheckpointWeight,
    hidden: usize,
    intermediate: usize,
}

#[derive(Clone, Debug)]
enum GgmlGateUp {
    Separate {
        gate: GgmlCheckpointWeight,
        up: GgmlCheckpointWeight,
    },
    /// One bank whose experts hold their gate rows, then their up rows
    /// (GGUF `ffn_gate_up_exps`).
    Halves(GgmlCheckpointWeight),
}

/// One projection of an expert bank: its blocks and where its rows sit.
#[derive(Clone, Copy, Debug)]
pub(in crate::backend) struct GgmlExpertBank<'a> {
    weight: &'a GgmlCheckpointWeight,
    rows: GgmlBankRows,
}

impl<'a> GgmlExpertBank<'a> {
    pub(in crate::backend) fn kind(&self) -> GgmlKernelType {
        self.weight.kind()
    }

    pub(in crate::backend) fn operand(&self) -> (&'a DeviceBuffer<u8>, GgmlBankRows) {
        (self.weight.blocks(), self.rows)
    }

    /// The bank's F32 output scales, one per expert.
    pub(in crate::backend) const fn scale(&self) -> Option<&'a DeviceBuffer<f32>> {
        self.weight.output_scale()
    }
}

impl GgmlExpertWeights {
    pub(in crate::backend) const fn gate(&self) -> GgmlExpertBank<'_> {
        self.half(0)
    }

    pub(in crate::backend) const fn up(&self) -> GgmlExpertBank<'_> {
        self.half(1)
    }

    pub(in crate::backend) const fn down(&self) -> GgmlExpertBank<'_> {
        GgmlExpertBank {
            weight: &self.down,
            rows: GgmlBankRows::whole(self.hidden),
        }
    }

    const fn half(&self, half: usize) -> GgmlExpertBank<'_> {
        let intermediate = self.intermediate;
        match &self.gate_up {
            GgmlGateUp::Separate { gate, up } => GgmlExpertBank {
                weight: if half == 0 {
                    gate
                } else {
                    up
                },
                rows: GgmlBankRows::whole(intermediate),
            },
            GgmlGateUp::Halves(gate_up) => GgmlExpertBank {
                weight: gate_up,
                rows: GgmlBankRows {
                    expert_rows: 2 * intermediate,
                    first_row: half * intermediate,
                },
            },
        }
    }
}

impl GgmlExpertWeights {
    /// The GGML banks of a dense-and-routed layer (Gemma 4), stacked
    /// separately or with gate and up in one bank of halves.
    pub(in crate::backend) fn load_hybrid(
        tensors: &CudaTensorSet,
        bindings: &HybridMoeExpertBindings<'_>,
        shape: (usize, usize, usize),
    ) -> Result<Option<Self>> {
        let routed = match bindings {
            HybridMoeExpertBindings::Stacked(banks) => RoutedExpertBindings::SeparateGateUp {
                gate: banks.gate,
                up: banks.up,
                down: banks.down,
            },
            HybridMoeExpertBindings::FusedStacked { gate_up, down } => {
                RoutedExpertBindings::InterleavedGateUp { gate_up, down }
            },
            HybridMoeExpertBindings::Individual { .. } => return Ok(None),
        };
        Self::load(tensors, routed, shape)
    }

    /// Multiplies every expert's down products by `factor`.
    pub(super) fn scale_down(&mut self, backend: &crate::CudaBackend, factor: f32) -> Result<()> {
        self.down = self.down.scaled(backend, factor)?;
        Ok(())
    }

    /// Experts of every bank.
    pub(in crate::backend) const fn experts(&self) -> usize {
        self.down.matrices()
    }

    /// The executor geometry of these banks.
    pub(in crate::backend) fn geometry(
        &self,
        experts: RoutedExpertsConfig,
        activation: crate::GatedActivation,
    ) -> GgmlExpertGeometry {
        GgmlExpertGeometry {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts,
            activation,
        }
    }

    /// Separate gate, up and down banks, for tests that build their blocks.
    #[cfg(test)]
    pub(crate) fn separate(
        [gate, up, down]: [GgmlCheckpointWeight; 3],
        (hidden, intermediate): (usize, usize),
    ) -> Self {
        Self {
            gate_up: GgmlGateUp::Separate { gate, up },
            down,
            hidden,
            intermediate,
        }
    }
}

impl GgmlExpertWeights {
    /// GGML gate, up and down banks, separate or with gate and up in one
    /// bank of halves, when all are GGML.
    pub(in crate::backend) fn load(
        tensors: &CudaTensorSet,
        bindings: RoutedExpertBindings<'_>,
        (experts, hidden, intermediate): (usize, usize, usize),
    ) -> Result<Option<Self>> {
        let ggml = |binding: &TensorBinding| matches!(binding.storage, TensorStorage::Ggml { .. });
        let (gate_up, down) = match bindings {
            RoutedExpertBindings::SeparateGateUp { gate, up, down }
                if [gate, up, down].iter().all(|binding| ggml(binding)) =>
            {
                let separate = GgmlGateUp::Separate {
                    gate: GgmlCheckpointWeight::load_binding(tensors, gate)?,
                    up: GgmlCheckpointWeight::load_binding(tensors, up)?,
                };
                (separate, down)
            },
            RoutedExpertBindings::InterleavedGateUp { gate_up, down }
                if ggml(gate_up)
                    && ggml(down)
                    && gate_up
                        .transforms
                        .contains(&BindingTransform::FusedGateUp { interleaved: false }) =>
            {
                (GgmlGateUp::Halves(GgmlCheckpointWeight::load_binding(tensors, gate_up)?), down)
            },
            _ => return Ok(None),
        };
        match &gate_up {
            GgmlGateUp::Separate { gate, up } => {
                gate.validate(hidden, experts * intermediate)?;
                up.validate(hidden, experts * intermediate)?;
            },
            GgmlGateUp::Halves(gate_up) => gate_up.validate(hidden, experts * 2 * intermediate)?,
        }
        let down = GgmlCheckpointWeight::load_binding(tensors, down)?;
        down.validate(intermediate, experts * hidden)?;
        Ok(Some(Self { gate_up, down, hidden, intermediate }))
    }
}
