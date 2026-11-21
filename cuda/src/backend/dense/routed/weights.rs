use mircuda::DeviceBuffer;
use models::weights::{
    BindingTransform, RoutedExpertBindings, RoutedFeedForwardBindings, TensorBinding, TensorStorage,
};

use super::super::DenseSwiGluConfig;
use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, GgmlCheckpointWeight, Result,
    backend::DenseExpertWeights,
    kernels::{GgmlBankRows, GgmlKernelType},
};

/// A layer's router and expert banks, bound at execution like the gated
/// MLP's projections so one executor serves every layer.
#[derive(Clone, Debug)]
pub struct RoutedExpertWeights {
    pub(super) router: CudaTensor,
    pub(super) experts: RoutedExperts,
}

#[derive(Clone, Debug)]
pub(super) enum RoutedExperts {
    Dense(Box<DenseExpertWeights>),
    Ggml(Box<GgmlExpertWeights>),
}

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

impl RoutedExpertWeights {
    pub(crate) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        bindings: RoutedFeedForwardBindings<'_>,
        config: DenseSwiGluConfig,
    ) -> Result<Self> {
        let experts = config
            .experts
            .ok_or(Error::InvalidExecutionPlan("dense layer config declares no routed experts"))?;
        let hidden = config.attention.hidden_size;
        let router = tensors
            .get(&bindings.router.source)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(bindings.router.source.clone()))?;
        if router.dtype() != CudaTensorDType::Bf16 || router.shape() != [experts.count, hidden] {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA routed experts need a dense [{}, {hidden}] router: {}",
                experts.count, bindings.router.source
            )));
        }
        let shape = (experts.count, hidden, config.intermediate_size);
        let banks = match GgmlExpertWeights::load(tensors, bindings.experts, shape)? {
            Some(weights) => RoutedExperts::Ggml(Box::new(weights)),
            None => RoutedExperts::Dense(Box::new(DenseExpertWeights::load(
                backend,
                tensors,
                bindings.experts,
                experts.count,
                hidden,
                config.intermediate_size,
            )?)),
        };
        Ok(Self { router, experts: banks })
    }
}
