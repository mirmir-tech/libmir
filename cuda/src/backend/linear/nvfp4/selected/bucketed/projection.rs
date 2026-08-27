use mircuda::{DeviceBuffer, Stream, VariableGroupedFp4Spec, bf16};

use super::{CudaBackend, NvFp4ExpertBank, buckets::ExpertBuckets, scratch::ProjectionScratch};
use crate::{
    Error, Result,
    kernels::{BucketQuantize, NvFp4BucketPreparation, NvFp4Spec},
};

#[derive(Debug)]
pub(super) struct BucketedNvFp4Projection {
    pub(super) bank: NvFp4ExpertBank,
    pub(super) stream: Stream,
    spec: NvFp4Spec,
    assignments: usize,
    experts: usize,
    tokens: usize,
    selected: usize,
}

impl BucketedNvFp4Projection {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let assignments = tokens
            .checked_mul(selected)
            .ok_or(Error::InvalidNvFp4("bucketed assignment count overflow"))?;
        let experts = bank.config.experts;
        if tokens == 0 || selected == 0 || selected > experts {
            return Err(Error::InvalidNvFp4("invalid bucketed expert geometry"));
        }
        let spec = NvFp4Spec::new(bank.config.input_features, bank.config.output_features)?;
        Ok(Self {
            bank,
            stream: backend.inner.stream.clone(),
            spec,
            assignments,
            experts,
            tokens,
            selected,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn quantize_pair(
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        left: &Self,
        right: &Self,
        left_scratch: &mut ProjectionScratch,
        right_scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        left.require_compatible(right)?;
        let geometry = left.quantize_geometry(false);
        preparation.quantize_pair(
            &left.stream,
            input,
            selected,
            &buckets.order,
            &buckets.offsets,
            &buckets.scale_offsets,
            &left.bank.input_scales,
            &right.bank.input_scales,
            &mut left_scratch.packed,
            &mut right_scratch.packed,
            &mut left_scratch.scales,
            &mut right_scratch.scales,
            geometry,
        )
    }

    pub(super) fn quantize_shared(
        &self,
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        scratch: &mut ProjectionScratch,
    ) -> Result<()> {
        let geometry = self.quantize_geometry(false);
        preparation.quantize(
            &self.stream,
            input,
            selected,
            &buckets.order,
            &buckets.offsets,
            &buckets.scale_offsets,
            &self.bank.input_scales,
            &mut scratch.packed,
            &mut scratch.scales,
            geometry,
        )
    }

    pub(super) fn plan_spec(&self) -> Result<VariableGroupedFp4Spec> {
        Ok(VariableGroupedFp4Spec::new(
            self.experts,
            self.experts,
            self.assignments,
            self.spec.output_features,
            self.spec.input_features,
            self.assignments,
        )?)
    }

    pub(super) fn output_elements(&self) -> Result<usize> {
        self.assignments
            .checked_mul(self.spec.output_features)
            .ok_or(Error::InvalidNvFp4("bucketed expert output overflow"))
    }

    pub(super) const fn output_features(&self) -> usize {
        self.spec.output_features
    }

    pub(super) const fn input_features(&self) -> usize {
        self.spec.input_features
    }

    pub(super) const fn tokens(&self) -> usize {
        self.tokens
    }

    fn require_compatible(&self, other: &Self) -> Result<()> {
        if self.spec.input_features == other.spec.input_features
            && self.spec.output_features == other.spec.output_features
            && self.assignments == other.assignments
            && self.experts == other.experts
            && self.tokens == other.tokens
            && self.selected == other.selected
        {
            Ok(())
        } else {
            Err(Error::InvalidNvFp4("incompatible paired bucket projections"))
        }
    }

    pub(super) const fn quantize_geometry(&self, ranked: bool) -> BucketQuantize {
        BucketQuantize {
            assignments: self.assignments,
            experts: self.experts,
            selected: self.selected,
            input_rows: if ranked {
                self.assignments
            } else {
                self.tokens
            },
            columns: self.spec.input_features,
            ranked,
        }
    }
}
