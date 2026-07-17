use mircuda::{DeviceBuffer, IndexedGroupedFp4Plan, IndexedGroupedFp4Spec, Stream, bf16};

use super::{CudaBackend, NvFp4ExpertBank};
use crate::{
    Error, GatedActivation, Result,
    kernels::{GroupedQuantize, NvFp4GroupedPreparation, NvFp4Spec, scale_elements},
};

mod pair;

pub(super) use pair::GroupedNvFp4PairBf16;

/// Device-indexed grouped W4A4 projection over one NVFP4 expert bank.
#[derive(Debug)]
pub(super) struct GroupedNvFp4LinearBf16 {
    preparation: NvFp4GroupedPreparation,
    plan: IndexedGroupedFp4Plan,
    bank: NvFp4ExpertBank,
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    stream: Stream,
    spec: NvFp4Spec,
    groups: usize,
    tokens: usize,
    selected: usize,
}

impl GroupedNvFp4LinearBf16 {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let groups = tokens
            .checked_mul(selected)
            .ok_or(Error::InvalidNvFp4("grouped expert count overflow"))?;
        if tokens == 0 || selected == 0 || selected > bank.config.experts {
            return Err(Error::InvalidNvFp4("invalid grouped expert geometry"));
        }
        let spec = NvFp4Spec::new(bank.config.input_features, bank.config.output_features)?;
        let packed_elements = groups
            .checked_mul(spec.input_features / 2)
            .ok_or(Error::InvalidNvFp4("grouped packed input overflow"))?;
        let scale_elements = groups
            .checked_mul(scale_elements(1, spec.input_features)?)
            .ok_or(Error::InvalidNvFp4("grouped input scale overflow"))?;
        let plan = IndexedGroupedFp4Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            IndexedGroupedFp4Spec::new(
                groups,
                bank.config.experts,
                1,
                spec.output_features,
                spec.input_features,
            )?,
        )?;
        Ok(Self {
            preparation: NvFp4GroupedPreparation::compile(&backend.inner.compiler)?,
            plan,
            bank,
            packed: backend.inner.pool.allocate::<u8>(&backend.inner.stream, packed_elements)?,
            scales: backend
                .inner
                .pool
                .allocate_zeroed::<u8>(&backend.inner.stream, scale_elements)?,
            stream: backend.inner.stream.clone(),
            spec,
            groups,
            tokens,
            selected,
        })
    }

    pub(super) fn quantize_pair(
        left: &mut Self,
        right: &mut Self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
    ) -> Result<()> {
        left.require_pair(right)?;
        let geometry = left.geometry(false);
        left.preparation.quantize_pair(
            &left.stream,
            input,
            selected,
            &left.bank.input_scales,
            &right.bank.input_scales,
            &mut left.packed,
            &mut right.packed,
            &mut left.scales,
            &mut right.scales,
            geometry,
        )
    }

    pub(super) fn execute_ranked(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.quantize(input, selected, true)?;
        self.execute_prepared(selected, output)
    }

    pub(super) fn execute_prepared(
        &mut self,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let expected = self.output_elements()?;
        if output.len() != expected {
            return Err(Error::QuantizedGemvLengthMismatch {
                operand: "grouped NVFP4 output",
                expected,
                actual: output.len(),
            });
        }
        Ok(self.plan.execute(
            &self.stream,
            &self.packed,
            &self.scales,
            &self.bank.weight,
            &self.bank.cutlass_scales,
            &self.bank.combined_scales,
            selected,
            output,
        )?)
    }

    pub(super) fn quantized_workspace(&mut self) -> (&mut DeviceBuffer<u8>, &mut DeviceBuffer<u8>) {
        (&mut self.packed, &mut self.scales)
    }

    pub(super) fn scale_stride(&self) -> Result<usize> {
        scale_elements(1, self.spec.input_features)
    }

    pub(super) fn quantize_gated(
        &mut self,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        activation: GatedActivation,
    ) -> Result<()> {
        let geometry = self.geometry(true);
        self.preparation.quantize_gated(
            &self.stream,
            gate,
            up,
            selected,
            &self.bank.input_scales,
            &mut self.packed,
            &mut self.scales,
            geometry,
            activation,
        )
    }

    pub(super) fn output_elements(&self) -> Result<usize> {
        self.groups
            .checked_mul(self.spec.output_features)
            .ok_or(Error::InvalidNvFp4("grouped expert output overflow"))
    }

    pub(super) const fn output_features(&self) -> usize {
        self.spec.output_features
    }

    pub(super) const fn input_features(&self) -> usize {
        self.spec.input_features
    }

    fn quantize(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        ranked: bool,
    ) -> Result<()> {
        let geometry = self.geometry(ranked);
        self.preparation.quantize(
            &self.stream,
            input,
            selected,
            &self.bank.input_scales,
            &mut self.packed,
            &mut self.scales,
            geometry,
        )
    }

    fn require_pair(&self, other: &Self) -> Result<()> {
        if self.spec.input_features == other.spec.input_features
            && self.groups == other.groups
            && self.tokens == other.tokens
            && self.selected == other.selected
        {
            Ok(())
        } else {
            Err(Error::InvalidNvFp4("incompatible paired grouped projections"))
        }
    }

    const fn geometry(&self, ranked: bool) -> GroupedQuantize {
        GroupedQuantize {
            groups: self.groups,
            selected: self.selected,
            input_rows: if ranked {
                self.groups
            } else {
                self.tokens
            },
            columns: self.spec.input_features,
            ranked,
        }
    }
}
