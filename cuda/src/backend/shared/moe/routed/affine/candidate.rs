use mircuda::{DeviceBuffer, bf16};

use super::super::super::{AffineSharedExpertMoeConfig, weights::AffineRoutedMoeWeights};
use crate::{
    AffineQuantizedConfig, AffineQuantizedPairTensors, CudaBackend, GatedActivation, Result,
    SelectedAffineGatedBf16Linear, SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear,
    backend::tuning::AffineMoeExecution, kernels::ElementwiseBf16,
};

#[derive(Debug)]
pub(super) struct Candidate {
    pub(super) execution: AffineMoeExecution,
    plan: Plan,
    activation: GatedActivation,
    stream: mircuda::Stream,
}

#[derive(Debug)]
enum Plan {
    Fused {
        gated: SelectedAffineGatedBf16Linear,
        down: SelectedAffineReduceBf16Linear,
    },
    Separate {
        pair: SelectedAffinePairBf16Linear,
        activation: ElementwiseBf16,
        gate: DeviceBuffer<bf16>,
        up: DeviceBuffer<bf16>,
        down: SelectedAffineReduceBf16Linear,
    },
}

impl Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        tokens: usize,
        execution: AffineMoeExecution,
    ) -> Result<Self> {
        let gate = AffineQuantizedConfig::new(
            config.hidden_size,
            config.routed_intermediate_size,
            config.group_size,
            config.expert_bits,
        );
        let down_config = AffineQuantizedConfig::new(
            config.routed_intermediate_size,
            config.hidden_size,
            config.group_size,
            config.expert_bits,
        );
        let down = backend.prepare_batched_selected_affine_reduce_bf16_linear(
            tokens,
            down_config,
            config.expert_count,
            config.top_k,
        )?;
        let plan = match execution {
            AffineMoeExecution::FusedGated => Plan::Fused {
                gated: backend.prepare_batched_selected_affine_gated_bf16_linear(
                    tokens,
                    gate,
                    config.expert_count,
                    config.top_k,
                    config.activation,
                )?,
                down,
            },
            AffineMoeExecution::SeparatePair => {
                let pair = backend.prepare_batched_selected_affine_pair_bf16_linear(
                    tokens,
                    gate,
                    config.expert_count,
                    config.top_k,
                )?;
                let elements = pair.output_elements()?;
                Plan::Separate {
                    pair,
                    activation: ElementwiseBf16::compile(backend.compiler(), elements)?,
                    gate: backend.pool().allocate(backend.stream(), elements)?,
                    up: backend.pool().allocate(backend.stream(), elements)?,
                    down,
                }
            },
        };
        Ok(Self {
            execution,
            plan,
            activation: config.activation,
            stream: backend.stream().clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &AffineRoutedMoeWeights,
        intermediate: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let pair = AffineQuantizedPairTensors {
            gate: weights.gate.tensors(),
            up: weights.up.tensors(),
        };
        match &mut self.plan {
            Plan::Fused { gated, down } => {
                gated.execute(input, selected, pair, intermediate)?;
                down.execute(intermediate, selected, routing, weights.down.tensors(), output)
            },
            Plan::Separate {
                pair: operation,
                activation,
                gate,
                up,
                down,
            } => {
                operation.execute(input, selected, pair, gate, up)?;
                activation.gated(&self.stream, gate, up, intermediate, self.activation.into())?;
                down.execute(intermediate, selected, routing, weights.down.tensors(), output)
            },
        }
    }
}
