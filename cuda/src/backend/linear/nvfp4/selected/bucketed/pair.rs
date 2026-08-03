use mircuda::{
    DeviceBuffer, PairedVariableGroupedFp4Launch, PairedVariableGroupedFp4Plan,
    VariableGroupedFp4Metadata, VariableGroupedFp4Operands, bf16,
};

use super::{
    BucketedNvFp4Projection, moe::ExpertBuckets, scratch::ProjectionScratch, validate_output,
};
use crate::{CudaBackend, NvFp4ExpertBank, Result, kernels::NvFp4BucketPreparation};

#[derive(Debug)]
pub(in crate::backend::linear::nvfp4::selected) struct BucketedNvFp4PairBf16 {
    plan: PairedVariableGroupedFp4Plan,
    left: BucketedNvFp4Projection,
    right: BucketedNvFp4Projection,
}

impl BucketedNvFp4PairBf16 {
    pub(in crate::backend::linear::nvfp4::selected) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        left_bank: NvFp4ExpertBank,
        right_bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let left = BucketedNvFp4Projection::new(backend, tokens, selected, left_bank)?;
        let right = BucketedNvFp4Projection::new(backend, tokens, selected, right_bank)?;
        let plan = PairedVariableGroupedFp4Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            left.plan_spec()?,
        )?;
        Ok(Self { plan, left, right })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::linear::nvfp4::selected) fn execute(
        &mut self,
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        left_output: &mut DeviceBuffer<bf16>,
        right_output: &mut DeviceBuffer<bf16>,
        left_scratch: &mut ProjectionScratch,
        right_scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        BucketedNvFp4Projection::quantize_pair(
            preparation, buckets, input, selected, &self.left, &self.right, left_scratch,
            right_scratch,
        )?;
        validate_output(&self.left, left_output)?;
        validate_output(&self.right, right_output)?;
        let Self { plan, left, right } = self;
        let mut launch = PairedVariableGroupedFp4Launch {
            left: operands(left, left_scratch, left_output),
            right: operands(right, right_scratch, right_output),
            metadata: VariableGroupedFp4Metadata {
                indices: &buckets.indices,
                rows: &buckets.counts,
                offsets: &buckets.offsets,
                scale_offsets: &buckets.scale_offsets,
            },
        };
        Ok(plan.execute(&left.stream, &mut launch)?)
    }

    pub(in crate::backend::linear::nvfp4::selected) fn output_elements(&self) -> Result<usize> {
        self.left.output_elements()
    }

    pub(in crate::backend::linear::nvfp4::selected) const fn output_features(&self) -> usize {
        self.left.output_features()
    }
}

fn operands<'a>(
    projection: &'a BucketedNvFp4Projection,
    scratch: &'a ProjectionScratch,
    output: &'a mut DeviceBuffer<bf16>,
) -> VariableGroupedFp4Operands<'a> {
    VariableGroupedFp4Operands {
        a: &scratch.packed,
        a_scales: &scratch.scales,
        b: &projection.bank.weight,
        b_scales: &projection.bank.cutlass_scales,
        alphas: &projection.bank.combined_scales,
        output,
    }
}
