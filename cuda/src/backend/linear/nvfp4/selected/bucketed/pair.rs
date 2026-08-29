use mircuda::{
    DeviceBuffer, PairedVariableGroupedFp4Launch, PairedVariableGroupedFp4Plan,
    VariableGroupedFp4Metadata, VariableGroupedFp4Operands, bf16,
};

use super::{
    BucketedNvFp4Projection, buckets::ExpertBuckets, scratch::ProjectionScratch, validate_output,
};
use crate::{
    CudaBackend, NvFp4ExpertBank, Result,
    kernels::{NvFp4BucketPreparation, NvFp4Preparation},
};

#[derive(Debug)]
pub(in crate::backend::linear::nvfp4::selected) struct BucketedNvFp4PairBf16 {
    plan: PairedVariableGroupedFp4Plan,
    left: BucketedNvFp4Projection,
    right: BucketedNvFp4Projection,
    shared_input: bool,
    uniform_input: bool,
    unique_quantization: NvFp4Preparation,
}

impl BucketedNvFp4PairBf16 {
    pub(in crate::backend::linear::nvfp4::selected) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        left_bank: NvFp4ExpertBank,
        right_bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let shared_input = left_bank.shares_input_quantization(&right_bank);
        let uniform_input = left_bank.shares_uniform_input_quantization(&right_bank);
        let left = BucketedNvFp4Projection::new(backend, tokens, selected, left_bank)?;
        let right = BucketedNvFp4Projection::new(backend, tokens, selected, right_bank)?;
        let plan = PairedVariableGroupedFp4Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            left.plan_spec()?,
        )?;
        Ok(Self {
            plan,
            left,
            right,
            shared_input,
            uniform_input,
            unique_quantization: NvFp4Preparation::compile(&backend.inner.compiler)?,
        })
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
        if self.uniform_input {
            self.unique_quantization.quantize(
                &self.left.stream,
                self.left.tokens(),
                self.left.input_features(),
                input,
                &self.left.bank.input_scales,
                &mut right_scratch.packed,
                &mut right_scratch.scales,
            )?;
            preparation.gather_quantized(
                &self.left.stream,
                selected,
                &buckets.order,
                &buckets.offsets,
                &buckets.scale_offsets,
                &right_scratch.packed,
                &right_scratch.scales,
                &mut left_scratch.packed,
                &mut left_scratch.scales,
                self.left.quantize_geometry(false),
            )?;
        } else if self.shared_input {
            self.left.quantize_shared(preparation, buckets, input, selected, left_scratch)?;
        } else {
            BucketedNvFp4Projection::quantize_pair(
                preparation, buckets, input, selected, &self.left, &self.right, left_scratch,
                right_scratch,
            )?;
        }
        self.launch(buckets, left_output, right_output, left_scratch, right_scratch)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::linear::nvfp4::selected) fn execute_prequantized(
        &mut self,
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        selected: &DeviceBuffer<u32>,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        left_output: &mut DeviceBuffer<bf16>,
        right_output: &mut DeviceBuffer<bf16>,
        left_scratch: &mut ProjectionScratch,
        right_scratch: &ProjectionScratch,
    ) -> Result<()> {
        preparation.gather_quantized(
            &self.left.stream,
            selected,
            &buckets.order,
            &buckets.offsets,
            &buckets.scale_offsets,
            packed,
            scales,
            &mut left_scratch.packed,
            &mut left_scratch.scales,
            self.left.quantize_geometry(false),
        )?;
        self.launch(buckets, left_output, right_output, left_scratch, right_scratch)
    }

    fn launch(
        &mut self,
        buckets: &ExpertBuckets,
        left_output: &mut DeviceBuffer<bf16>,
        right_output: &mut DeviceBuffer<bf16>,
        left_scratch: &ProjectionScratch,
        right_scratch: &ProjectionScratch,
    ) -> Result<()> {
        validate_output(&self.left, left_output)?;
        validate_output(&self.right, right_output)?;
        let right_input = if self.shared_input {
            left_scratch
        } else {
            right_scratch
        };
        let Self { plan, left, right, .. } = self;
        let mut launch = PairedVariableGroupedFp4Launch {
            left: operands(left, left_scratch, left_output),
            right: operands(right, right_input, right_output),
            metadata: VariableGroupedFp4Metadata {
                indices: &buckets.indices,
                rows: &buckets.counts,
                offsets: &buckets.offsets,
                scale_offsets: &buckets.scale_offsets,
            },
        };
        Ok(plan.execute(&left.stream, &mut launch)?)
    }

    pub(in crate::backend::linear::nvfp4::selected) fn prequant_scale(
        &self,
    ) -> Option<DeviceBuffer<f32>> {
        self.uniform_input.then(|| self.left.bank.input_scales.clone())
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
