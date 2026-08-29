use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    super::{CudaBackend, NvFp4ExpertBank},
    BucketedNvFp4LinearBf16, BucketedNvFp4PairBf16,
    scratch::{BucketedNvFp4Scratch, BucketedNvFp4ScratchConfig},
};
use crate::{
    Error, GatedActivation, Result,
    kernels::{BucketGeometry, ElementwiseBf16, NvFp4BucketPreparation},
};

/// Prefill-oriented NVFP4 `MoE` with device-side token bucketing per expert.
#[derive(Debug)]
pub struct BucketedNvFp4MoeBf16 {
    preparation: NvFp4BucketPreparation,
    pub(super) gate_up: BucketedNvFp4PairBf16,
    down: BucketedNvFp4LinearBf16,
    reduce: ElementwiseBf16,
    scratch: Arc<Mutex<BucketedNvFp4Scratch>>,
    activation: GatedActivation,
    stream: Stream,
    tokens: usize,
    selected: usize,
    experts: usize,
    assignments: usize,
    pub(super) output_elements: usize,
}

#[derive(Clone, Copy)]
enum BucketedInput<'a> {
    Bf16(&'a DeviceBuffer<bf16>),
    NvFp4 {
        packed: &'a DeviceBuffer<u8>,
        scales: &'a DeviceBuffer<u8>,
    },
}

enum BucketedOutput<'a> {
    Routed(&'a mut DeviceBuffer<bf16>),
    ResidualShared {
        residual: &'a DeviceBuffer<bf16>,
        shared: &'a DeviceBuffer<bf16>,
        output: &'a mut DeviceBuffer<bf16>,
    },
}

impl BucketedNvFp4MoeBf16 {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate_bank: NvFp4ExpertBank,
        up_bank: NvFp4ExpertBank,
        down_bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let experts = gate_bank.config.experts;
        let assignments = tokens
            .checked_mul(selected)
            .ok_or(Error::InvalidNvFp4("bucketed assignment count overflow"))?;
        let gate_up = BucketedNvFp4PairBf16::new(backend, tokens, selected, gate_bank, up_bank)?;
        let down = BucketedNvFp4LinearBf16::new(backend, tokens, selected, down_bank)?;
        if down.output_features() == 0 || gate_up.output_features() != down.input_features() {
            return Err(Error::InvalidNvFp4("incompatible bucketed expert banks"));
        }
        let output_elements = tokens
            .checked_mul(down.output_features())
            .ok_or(Error::InvalidNvFp4("bucketed MoE output overflow"))?;
        tracing::debug!(
            backend = "cuda",
            mode = "device_bucketed",
            tokens,
            top_k = selected,
            experts,
            assignments,
            "prepared NVFP4 MoE execution"
        );
        let scratch = backend.bucketed_nvfp4_scratch(BucketedNvFp4ScratchConfig {
            tokens,
            selected,
            experts,
            hidden: down.output_features(),
            intermediate: gate_up.output_features(),
        })?;
        Ok(Self {
            preparation: NvFp4BucketPreparation::compile(&backend.inner.compiler)?,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, down.output_features())?,
            scratch,
            gate_up,
            down,
            activation,
            stream: backend.inner.stream.clone(),
            tokens,
            selected,
            experts,
            assignments,
            output_elements,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_input(
            BucketedInput::Bf16(input),
            selected,
            routing,
            BucketedOutput::Routed(output),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_prequantized_residual_shared(
        &mut self,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        residual: &DeviceBuffer<bf16>,
        shared: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_input(
            BucketedInput::NvFp4 { packed, scales },
            selected,
            routing,
            BucketedOutput::ResidualShared { residual, shared, output },
        )
    }

    fn execute_input(
        &mut self,
        input: BucketedInput<'_>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: BucketedOutput<'_>,
    ) -> Result<()> {
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("NVFP4 bucket scratch lock is poisoned"))?;
        let BucketedNvFp4Scratch {
            buckets,
            gate,
            up,
            down,
            gate_output,
            up_output,
            down_output,
        } = &mut *scratch;
        self.preparation.prepare(
            &self.stream,
            selected,
            &mut buckets.counts,
            &mut buckets.offsets,
            &mut buckets.scale_offsets,
            &mut buckets.order,
            &mut buckets.positions,
            &mut buckets.indices,
            BucketGeometry {
                assignments: self.assignments,
                experts: self.experts,
            },
        )?;
        match input {
            BucketedInput::Bf16(input) => self.gate_up.execute(
                &self.preparation,
                buckets,
                input,
                selected,
                gate_output,
                up_output,
                gate,
                up,
            )?,
            BucketedInput::NvFp4 { packed, scales } => self.gate_up.execute_prequantized(
                &self.preparation,
                buckets,
                selected,
                packed,
                scales,
                gate_output,
                up_output,
                gate,
                up,
            )?,
        }
        self.down.execute_gated(
            &self.preparation,
            buckets,
            gate_output,
            up_output,
            selected,
            self.activation,
            down_output,
            down,
        )?;
        let result = match output {
            BucketedOutput::Routed(output) => self.reduce.weighted_reduce_bucketed(
                &self.stream,
                down_output,
                routing,
                &buckets.positions,
                output,
                self.selected,
                self.tokens,
            ),
            BucketedOutput::ResidualShared { residual, shared, output } => {
                self.reduce.weighted_reduce_bucketed_residual_shared(
                    &self.stream,
                    down_output,
                    routing,
                    &buckets.positions,
                    residual,
                    shared,
                    output,
                    self.selected,
                    self.tokens,
                )
            },
        };
        drop(scratch);
        result
    }
}
