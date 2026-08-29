mod api;
mod buckets;
mod moe;
mod pair;
mod projection;
mod scratch;

use mircuda::{DeviceBuffer, VariableGroupedFp4Plan, bf16};
pub use moe::BucketedNvFp4MoeBf16;
pub(super) use pair::BucketedNvFp4PairBf16;
use projection::BucketedNvFp4Projection;
use scratch::ProjectionScratch;
pub(in crate::backend) use scratch::{BucketedNvFp4Scratch, BucketedNvFp4ScratchConfig};

use self::buckets::ExpertBuckets;
use super::{CudaBackend, NvFp4ExpertBank};
use crate::{Error, GatedActivation, Result, kernels::NvFp4BucketPreparation};

/// Expert-bucketed W4A4 projection with variable rows per CUTLASS group.
#[derive(Debug)]
pub(super) struct BucketedNvFp4LinearBf16 {
    plan: VariableGroupedFp4Plan,
    projection: BucketedNvFp4Projection,
}

impl BucketedNvFp4LinearBf16 {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let projection = BucketedNvFp4Projection::new(backend, tokens, selected, bank)?;
        let plan = VariableGroupedFp4Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            projection.plan_spec()?,
        )?;
        Ok(Self { plan, projection })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_gated(
        &mut self,
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        activation: GatedActivation,
        output: &mut DeviceBuffer<bf16>,
        scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        preparation.quantize_gated(
            &self.projection.stream,
            gate,
            up,
            selected,
            &buckets.order,
            &buckets.offsets,
            &buckets.scale_offsets,
            &self.projection.bank.input_scales,
            &mut scratch.packed,
            &mut scratch.scales,
            self.projection.quantize_geometry(true),
            activation,
        )?;
        self.execute_prepared(buckets, output, scratch)
    }

    fn execute_prepared(
        &mut self,
        buckets: &ExpertBuckets,
        output: &mut DeviceBuffer<bf16>,
        scratch: &ProjectionScratch,
    ) -> Result<()> {
        validate_output(&self.projection, output)?;
        let projection = &self.projection;
        Ok(self.plan.execute(
            &projection.stream,
            &scratch.packed,
            &scratch.scales,
            &projection.bank.weight,
            &projection.bank.cutlass_scales,
            &projection.bank.combined_scales,
            &buckets.indices,
            &buckets.counts,
            &buckets.offsets,
            &buckets.scale_offsets,
            output,
        )?)
    }

    pub(super) const fn output_features(&self) -> usize {
        self.projection.output_features()
    }

    pub(super) const fn input_features(&self) -> usize {
        self.projection.input_features()
    }
}

fn validate_output(
    projection: &BucketedNvFp4Projection,
    output: &DeviceBuffer<bf16>,
) -> Result<()> {
    let expected = projection.output_elements()?;
    if output.len() == expected {
        Ok(())
    } else {
        Err(Error::QuantizedGemvLengthMismatch {
            operand: "bucketed NVFP4 output",
            expected,
            actual: output.len(),
        })
    }
}
