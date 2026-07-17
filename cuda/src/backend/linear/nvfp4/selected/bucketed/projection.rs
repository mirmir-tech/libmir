use mircuda::{DeviceBuffer, Stream, VariableGroupedFp4Spec, bf16};

use super::super::{CudaBackend, NvFp4ExpertBank, bucketed_moe::ExpertBuckets};
use crate::{
    Error, Result,
    kernels::{BucketQuantize, NvFp4BucketPreparation, NvFp4Spec, scale_elements},
};

#[derive(Debug)]
pub(super) struct BucketedNvFp4Projection {
    pub(super) bank: NvFp4ExpertBank,
    pub(super) packed: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
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
        let packed_elements = assignments
            .checked_mul(spec.input_features / 2)
            .ok_or(Error::InvalidNvFp4("bucketed packed input overflow"))?;
        let scale_count = experts
            .checked_mul(scale_elements(assignments, spec.input_features)?)
            .ok_or(Error::InvalidNvFp4("bucketed input scale overflow"))?;
        Ok(Self {
            bank,
            packed: backend.inner.pool.allocate(&backend.inner.stream, packed_elements)?,
            scales: backend.inner.pool.allocate_zeroed(&backend.inner.stream, scale_count)?,
            stream: backend.inner.stream.clone(),
            spec,
            assignments,
            experts,
            tokens,
            selected,
        })
    }

    pub(super) fn quantize_pair(
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        left: &mut Self,
        right: &mut Self,
    ) -> Result<()> {
        left.require_compatible(right)?;
        let geometry = left.quantize_geometry(false);
        preparation.quantize_pair(
            &left.stream,
            input,
            selected,
            &buckets.order,
            &buckets.offsets,
            &left.bank.input_scales,
            &right.bank.input_scales,
            &mut left.packed,
            &mut right.packed,
            &mut left.scales,
            &mut right.scales,
            geometry,
        )
    }

    pub(super) fn quantize_ranked(
        &mut self,
        preparation: &NvFp4BucketPreparation,
        buckets: &ExpertBuckets,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let geometry = self.quantize_geometry(true);
        preparation.quantize(
            &self.stream,
            input,
            selected,
            &buckets.order,
            &buckets.offsets,
            &self.bank.input_scales,
            &mut self.packed,
            &mut self.scales,
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

    const fn quantize_geometry(&self, ranked: bool) -> BucketQuantize {
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
