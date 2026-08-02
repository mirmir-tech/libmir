use mircuda::{DeviceBuffer, Stream, bf16};

use super::{MxFp8ExpertWeights, MxFp8GatheredBf16Linear, weights::MxFp8GateUpWeights};
use crate::{CudaBackend, Error, GatedActivation, Result, kernels::ElementwiseBf16};

mod tuning;

#[derive(Debug)]
struct Candidate {
    execution: crate::backend::tuning::MxFp8MoeExecution,
    gate_up: CandidateGateUp,
    down: MxFp8GatheredBf16Linear,
}

#[derive(Debug)]
enum CandidateGateUp {
    Separate {
        gate: MxFp8GatheredBf16Linear,
        up: MxFp8GatheredBf16Linear,
    },
    Interleaved {
        gate_up: MxFp8GatheredBf16Linear,
    },
}

#[derive(Debug)]
enum GateUpScratch {
    Separate {
        gate: DeviceBuffer<bf16>,
        up: DeviceBuffer<bf16>,
    },
    Interleaved {
        gate_up: DeviceBuffer<bf16>,
    },
}

#[derive(Debug)]
/// Autotuned gathered MXFP8 routed MLP with accelerator-resident routing.
pub struct MxFp8GatheredMoeBf16 {
    candidates: Vec<Candidate>,
    activation: crate::kernels::GatedActivation,
    gated: ElementwiseBf16,
    reduce: ElementwiseBf16,
    gate_up_output: GateUpScratch,
    activated: DeviceBuffer<bf16>,
    routed_output: DeviceBuffer<bf16>,
    stream: Stream,
    tokens: usize,
    selected_count: usize,
}

impl MxFp8GatheredMoeBf16 {
    pub fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected_count: usize,
        activation: GatedActivation,
        weights: &MxFp8ExpertWeights,
    ) -> Result<Self> {
        tuning::prepare(backend, tokens, selected_count, activation, weights)
    }

    fn with_candidates(
        backend: &CudaBackend,
        tokens: usize,
        selected_count: usize,
        activation: GatedActivation,
        weights: &MxFp8ExpertWeights,
        executions: &[crate::backend::tuning::MxFp8MoeExecution],
    ) -> Result<Self> {
        let assignments = tokens
            .checked_mul(selected_count)
            .ok_or(Error::InvalidDecoderKernel("MXFP8 expert assignment size overflow"))?;
        let intermediate_elements = weights.intermediate_elements(assignments)?;
        let routed_output_elements = weights.routed_output_elements(assignments)?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        let candidates = executions
            .iter()
            .copied()
            .map(|execution| Candidate::new(backend, tokens, selected_count, weights, execution))
            .collect::<Result<Vec<_>>>()?;
        let gate_up_output =
            match weights.gate_up {
                MxFp8GateUpWeights::Separate { .. } => GateUpScratch::Separate {
                    gate: allocate(intermediate_elements)?,
                    up: allocate(intermediate_elements)?,
                },
                MxFp8GateUpWeights::Interleaved { .. } => GateUpScratch::Interleaved {
                    gate_up: allocate(intermediate_elements.checked_mul(2).ok_or(
                        Error::InvalidDecoderKernel("MXFP8 fused gate/up size overflow"),
                    )?)?,
                },
            };
        Ok(Self {
            candidates,
            activation: activation.into(),
            gated: ElementwiseBf16::compile(&backend.inner.compiler, intermediate_elements)?,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, weights.hidden)?,
            gate_up_output,
            activated: allocate(intermediate_elements)?,
            routed_output: allocate(routed_output_elements)?,
            stream: backend.inner.stream.clone(),
            tokens,
            selected_count,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &MxFp8ExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_candidate(0, input, selected, routing, weights, output)
    }

    fn execute_candidate(
        &mut self,
        index: usize,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &MxFp8ExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let candidate = &self.candidates[index];
        match (&candidate.gate_up, &weights.gate_up, &mut self.gate_up_output) {
            (
                CandidateGateUp::Separate { gate, up },
                MxFp8GateUpWeights::Separate { gate: gate_weight, up: up_weight },
                GateUpScratch::Separate { gate: gate_output, up: up_output },
            ) => {
                gate.execute(input, selected, gate_weight, gate_output)?;
                up.execute(input, selected, up_weight, up_output)?;
                self.gated.gated(
                    &self.stream,
                    gate_output,
                    up_output,
                    &mut self.activated,
                    self.activation,
                )?;
            },
            (
                CandidateGateUp::Interleaved { gate_up },
                MxFp8GateUpWeights::Interleaved { gate_up: weight },
                GateUpScratch::Interleaved { gate_up: output },
            ) => {
                gate_up.execute(input, selected, weight, output)?;
                self.gated.gated_interleaved(
                    &self.stream,
                    output,
                    &mut self.activated,
                    weights.intermediate,
                    self.activation,
                )?;
            },
            _ => return Err(Error::InvalidExecutionPlan("MXFP8 gate/up storage changed")),
        }
        candidate.down.execute(
            &self.activated,
            selected,
            &weights.down,
            &mut self.routed_output,
        )?;
        self.reduce.weighted_reduce_batch(
            &self.stream,
            &self.routed_output,
            routing,
            output,
            self.selected_count,
            self.tokens,
        )
    }

    fn retain(&mut self, index: usize) {
        let selected = self.candidates.swap_remove(index);
        self.candidates.clear();
        self.candidates.push(selected);
    }
}

impl Candidate {
    fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected_count: usize,
        weights: &MxFp8ExpertWeights,
        execution: crate::backend::tuning::MxFp8MoeExecution,
    ) -> Result<Self> {
        let warps = execution.warps_per_block();
        let assignments = tokens
            .checked_mul(selected_count)
            .ok_or(Error::InvalidDecoderKernel("MXFP8 expert assignment size overflow"))?;
        let gate_up = match &weights.gate_up {
            MxFp8GateUpWeights::Separate { gate, up } => CandidateGateUp::Separate {
                gate: gate.prepare_gathered_routed_warps(backend, tokens, selected_count, warps)?,
                up: up.prepare_gathered_routed_warps(backend, tokens, selected_count, warps)?,
            },
            MxFp8GateUpWeights::Interleaved { gate_up } => CandidateGateUp::Interleaved {
                gate_up: gate_up
                    .prepare_gathered_routed_warps(backend, tokens, selected_count, warps)?,
            },
        };
        Ok(Self {
            execution,
            gate_up,
            down: weights.down.prepare_gathered_warps(backend, assignments, warps)?,
        })
    }
}
