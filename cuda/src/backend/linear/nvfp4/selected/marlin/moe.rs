use std::sync::{Arc, Mutex};

use mircuda::{
    Context, DeviceBuffer, MarlinNvFp4MoeOperands, MarlinNvFp4MoeSpec, MarlinNvFp4ThreadConfig,
    Stream, bf16,
};

use super::scratch::{MarlinNvFp4Scratch, MarlinNvFp4ScratchConfig};
use crate::{
    CudaBackend, Error, GatedActivation, NvFp4ExpertBank, Result,
    backend::linear::nvfp4::bank::MarlinNvFp4Bank, kernels::ElementwiseBf16,
};

/// Block-8 Marlin W4A16 execution for sparse decode batches.
#[derive(Debug)]
pub(in crate::backend) struct MarlinNvFp4MoeBf16 {
    gate_up: MarlinNvFp4Bank,
    down: MarlinNvFp4Bank,
    scratch: Arc<Mutex<MarlinNvFp4Scratch>>,
    gated: ElementwiseBf16,
    reduce: ElementwiseBf16,
    activation: GatedActivation,
    context: Context,
    stream: Stream,
    tokens: usize,
    top_k: usize,
    thread_config: MarlinNvFp4ThreadConfig,
}

impl CudaBackend {
    pub(in crate::backend) fn prepare_marlin_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        top_k: usize,
        activation: GatedActivation,
        thread_config: MarlinNvFp4ThreadConfig,
        banks: [NvFp4ExpertBank; 3],
    ) -> Result<MarlinNvFp4MoeBf16> {
        MarlinNvFp4MoeBf16::new(self, tokens, top_k, activation, thread_config, banks)
    }
}

impl MarlinNvFp4MoeBf16 {
    fn new(
        backend: &CudaBackend,
        tokens: usize,
        top_k: usize,
        activation: GatedActivation,
        thread_config: MarlinNvFp4ThreadConfig,
        [gate, up, down]: [NvFp4ExpertBank; 3],
    ) -> Result<Self> {
        let gate_up = gate.marlin_pair(backend, &up)?;
        let down = down.marlin(backend)?;
        let experts = gate_up.config.experts;
        let hidden = gate_up.config.input_features;
        let intermediate = down.config.input_features;
        if gate_up.config.output_features != intermediate * 2
            || down.config.output_features != hidden
            || down.config.experts != experts
            || tokens == 0
            || top_k == 0
        {
            return Err(Error::InvalidNvFp4("incompatible Marlin MoE banks"));
        }
        let assignments = product(tokens, top_k)?;
        Ok(Self {
            gate_up,
            down,
            scratch: backend.marlin_nvfp4_scratch(MarlinNvFp4ScratchConfig {
                tokens,
                top_k,
                experts,
                hidden,
                intermediate,
            })?,
            gated: ElementwiseBf16::compile(
                &backend.inner.compiler,
                product(assignments, intermediate)?,
            )?,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, hidden)?,
            activation,
            context: backend.inner.context.clone(),
            stream: backend.inner.stream.clone(),
            tokens,
            top_k,
            thread_config,
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("Marlin scratch lock is poisoned"))?;
        let experts = self.gate_up.config.experts;
        let hidden = self.gate_up.config.input_features;
        let intermediate = self.down.config.input_features;
        let first = MarlinNvFp4MoeSpec::new(
            experts,
            self.tokens,
            self.top_k,
            intermediate * 2,
            hidden,
            self.thread_config,
        )?;
        prepare_routes(&self.context, &self.stream, first, selected, routing, &mut scratch)?;
        execute_projection(
            &self.context,
            &self.stream,
            first,
            input,
            &self.gate_up,
            &mut scratch,
            ProjectionOutput::GateUp,
        )?;
        let gate_up_output = scratch.gate_up.clone();
        self.gated.gated_concatenated(
            &self.stream,
            &gate_up_output,
            &mut scratch.intermediate,
            intermediate,
            self.activation.into(),
        )?;
        let second = MarlinNvFp4MoeSpec::new(
            experts,
            first.assignments(),
            1,
            hidden,
            intermediate,
            self.thread_config,
        )?;
        let intermediate_input = scratch.intermediate.clone();
        execute_projection(
            &self.context,
            &self.stream,
            second,
            &intermediate_input,
            &self.down,
            &mut scratch,
            ProjectionOutput::Down,
        )?;
        self.reduce.weighted_reduce_batch(
            &self.stream, &scratch.down, routing, output, self.top_k, self.tokens,
        )
    }
}

fn prepare_routes(
    context: &Context,
    stream: &Stream,
    spec: MarlinNvFp4MoeSpec,
    selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>,
    scratch: &mut MarlinNvFp4Scratch,
) -> Result<()> {
    Ok(context.marlin_prepare_moe_routes(
        stream,
        spec,
        selected,
        routing,
        &mut scratch.sorted,
        &mut scratch.expert_ids,
        &mut scratch.padded,
        &mut scratch.offsets,
        &mut scratch.routing,
    )?)
}

#[derive(Clone, Copy)]
enum ProjectionOutput {
    GateUp,
    Down,
}

fn execute_projection(
    context: &Context,
    stream: &Stream,
    spec: MarlinNvFp4MoeSpec,
    input: &DeviceBuffer<bf16>,
    bank: &MarlinNvFp4Bank,
    scratch: &mut MarlinNvFp4Scratch,
    destination: ProjectionOutput,
) -> Result<()> {
    let output = match destination {
        ProjectionOutput::GateUp => &mut scratch.gate_up,
        ProjectionOutput::Down => &mut scratch.down,
    };
    Ok(context.marlin_nvfp4_moe(
        stream,
        spec,
        &MarlinNvFp4MoeOperands {
            input,
            weight: &bank.weight,
            scales: &bank.scales,
            global_scales: &bank.global_scales,
            routing: &scratch.routing,
            sorted: &scratch.sorted,
            expert_ids: &scratch.expert_ids,
            padded: &scratch.padded,
            temporary: &mut scratch.temporary,
            locks: &mut scratch.locks,
            output,
            multiply_routing: false,
            atomic_reduce: false,
        },
    )?)
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::InvalidNvFp4("Marlin MoE size overflow"))
}
