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

#[derive(Debug)]
pub(in crate::backend::linear::nvfp4::selected) struct ExpertBuckets {
    pub(super) counts: DeviceBuffer<u32>,
    pub(super) offsets: DeviceBuffer<u32>,
    pub(super) scale_offsets: DeviceBuffer<u32>,
    pub(super) order: DeviceBuffer<u32>,
    pub(super) positions: DeviceBuffer<u32>,
    pub(super) indices: DeviceBuffer<u32>,
}

impl ExpertBuckets {
    pub(super) fn new(backend: &CudaBackend, assignments: usize, experts: usize) -> Result<Self> {
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            counts: allocate(experts)?,
            offsets: allocate(experts)?,
            scale_offsets: allocate(experts)?,
            order: allocate(assignments)?,
            positions: allocate(assignments)?,
            indices: allocate(experts)?,
        })
    }
}

/// Prefill-oriented NVFP4 `MoE` with device-side token bucketing per expert.
#[derive(Debug)]
pub struct BucketedNvFp4MoeBf16 {
    preparation: NvFp4BucketPreparation,
    gate_up: BucketedNvFp4PairBf16,
    down: BucketedNvFp4LinearBf16,
    gated: ElementwiseBf16,
    reduce: ElementwiseBf16,
    scratch: Arc<Mutex<BucketedNvFp4Scratch>>,
    activation: GatedActivation,
    stream: Stream,
    tokens: usize,
    selected: usize,
    experts: usize,
    assignments: usize,
    output_elements: usize,
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn prepare_bucketed_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<BucketedNvFp4MoeBf16> {
        BucketedNvFp4MoeBf16::new(self, tokens, selected, activation, gate, up, down)
    }
}

impl BucketedNvFp4MoeBf16 {
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        let intermediate_elements = gate_up.output_elements()?;
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
            gated: ElementwiseBf16::compile(&backend.inner.compiler, intermediate_elements)?,
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
            intermediate,
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
        self.gate_up.execute(
            &self.preparation,
            buckets,
            input,
            selected,
            gate_output,
            up_output,
            gate,
            up,
        )?;
        self.gated.gated(
            &self.stream,
            gate_output,
            up_output,
            intermediate,
            self.activation.into(),
        )?;
        self.down.execute_ranked(
            &self.preparation,
            buckets,
            intermediate,
            selected,
            down_output,
            down,
        )?;
        let result = self.reduce.weighted_reduce_bucketed(
            &self.stream,
            down_output,
            routing,
            &buckets.positions,
            output,
            self.selected,
            self.tokens,
        );
        drop(scratch);
        result
    }

    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.output_elements
    }
}
