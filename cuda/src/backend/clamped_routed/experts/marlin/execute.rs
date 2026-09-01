use mircuda::{
    Context, DeviceBuffer, MarlinMxFp4MoeOperands, MarlinNvFp4MoeSpec, MarlinNvFp4ThreadConfig,
    Stream, bf16,
};

use super::MarlinMxFp4Candidate;
use crate::{
    CudaTensor, Error, Result,
    backend::{
        clamped_routed::weights::{NativeExpertWeights, marlin::MarlinMxFp4Bank},
        linear::MarlinNvFp4Scratch,
    },
};

impl MarlinMxFp4Candidate {
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the scratch lock makes the complete asynchronous CUDA enqueue atomic"
    )]
    pub(in crate::backend::clamped_routed::experts) fn execute(
        &self,
        weights: &NativeExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("MXFP4 Marlin scratch is poisoned"))?;
        let first = self.spec(
            self.tokens,
            self.config.top_k,
            self.banks.gate_up.output_features,
            self.banks.gate_up.input_features,
            self.gate_thread_config,
        )?;
        prepare_routes(&self.context, &self.stream, first, selected, routing, &mut scratch)?;
        self.epilogue
            .pad_input(&self.stream, input, &mut scratch.padded_input, self.tokens)?;
        let padded_input = scratch.padded_input.clone();
        execute_projection(
            &self.context,
            &self.stream,
            first,
            &padded_input,
            &self.banks.gate_up,
            &mut scratch,
            Projection::GateUp,
        )?;
        let gate_up = scratch.gate_up.clone();
        self.epilogue.gate_up(
            &self.stream,
            &gate_up,
            bf16s(&weights.gate_up_bias)?,
            selected,
            &mut scratch.padded_intermediate,
        )?;
        let assignments = self.tokens * self.config.top_k;
        let second = self.spec(
            assignments,
            1,
            self.banks.down.output_features,
            self.banks.down.input_features,
            self.down_thread_config,
        )?;
        let intermediate = scratch.padded_intermediate.clone();
        execute_projection(
            &self.context,
            &self.stream,
            second,
            &intermediate,
            &self.banks.down,
            &mut scratch,
            Projection::Down,
        )?;
        let padded_down = scratch.padded_down.clone();
        self.epilogue.down_reduce(
            &self.stream,
            &padded_down,
            bf16s(&weights.down_bias)?,
            selected,
            routing,
            output,
        )
    }

    fn spec(
        &self,
        tokens: usize,
        top_k: usize,
        n: usize,
        k: usize,
        thread_config: MarlinNvFp4ThreadConfig,
    ) -> Result<MarlinNvFp4MoeSpec> {
        Ok(MarlinNvFp4MoeSpec::new(
            self.config.experts,
            tokens,
            top_k,
            n,
            k,
            thread_config,
        )?)
    }
}

#[derive(Clone, Copy)]
enum Projection {
    GateUp,
    Down,
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

fn execute_projection(
    context: &Context,
    stream: &Stream,
    spec: MarlinNvFp4MoeSpec,
    input: &DeviceBuffer<bf16>,
    bank: &MarlinMxFp4Bank,
    scratch: &mut MarlinNvFp4Scratch,
    projection: Projection,
) -> Result<()> {
    let output = match projection {
        Projection::GateUp => &mut scratch.gate_up,
        Projection::Down => &mut scratch.padded_down,
    };
    Ok(context.marlin_mxfp4_moe(
        stream,
        spec,
        &MarlinMxFp4MoeOperands {
            input,
            weight: &bank.weight,
            scales: &bank.scales,
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

fn bf16s(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
