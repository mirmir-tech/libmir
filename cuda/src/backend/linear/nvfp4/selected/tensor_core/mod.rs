mod moe;
#[cfg(test)]
mod tests;
mod validation;

use mircuda::{BlockScaledFp4Plan, BlockScaledFp4Spec, DeviceBuffer, Stream, bf16};
pub use moe::SelectedNvFp4TensorCoreMoeBf16;
use validation::require_len;

use super::{CudaBackend, NvFp4ExpertBank};
use crate::{
    Error, Result,
    kernels::{
        NvFp4SelectedWeightLaunch, NvFp4SelectedWeightPreparation, NvFp4Spec, scale_elements,
    },
};

/// Device-selected NVFP4 projections executed by native W4A4 Tensor Cores.
#[derive(Debug)]
pub struct SelectedNvFp4LinearBf16 {
    selected_preparation: NvFp4SelectedWeightPreparation,
    bank: NvFp4ExpertBank,
    ranks: Vec<RankWorkspace>,
    stream: Stream,
    spec: NvFp4Spec,
}

#[derive(Debug)]
struct RankWorkspace {
    plan: BlockScaledFp4Plan,
    weight: DeviceBuffer<u8>,
    weight_scales: DeviceBuffer<u8>,
    input_global: DeviceBuffer<f32>,
    weight_global: DeviceBuffer<f32>,
    input: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<u8>,
    raw_output: DeviceBuffer<bf16>,
}

impl CudaBackend {
    /// Prepares a fixed-top-k Tensor Core projection over an NVFP4 expert bank.
    pub fn prepare_selected_nvfp4_linear_bf16(
        &self,
        selected: usize,
        bank: NvFp4ExpertBank,
    ) -> Result<SelectedNvFp4LinearBf16> {
        SelectedNvFp4LinearBf16::new(self, selected, bank)
    }
}

impl SelectedNvFp4LinearBf16 {
    pub(super) fn new(
        backend: &CudaBackend,
        selected: usize,
        bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        if selected == 0 || selected > bank.config.experts {
            return Err(Error::InvalidNvFp4("invalid selected expert count"));
        }
        let spec = NvFp4Spec::new(bank.config.input_features, bank.config.output_features)?;
        let selected_preparation =
            NvFp4SelectedWeightPreparation::compile(&backend.inner.compiler)?;
        let ranks = (0..selected)
            .map(|_| RankWorkspace::new(backend, spec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            selected_preparation,
            bank,
            ranks,
            stream: backend.inner.stream.clone(),
            spec,
        })
    }

    /// Enqueues gather, activation quantization, W4A4 matmul, and scaling.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require_len("selected NVFP4 input", self.spec.input_features, input.len())?;
        require_len("selected NVFP4 indices", self.ranks.len(), selected.len())?;
        let output_elements = self.output_elements()?;
        require_len("selected NVFP4 output", output_elements, output.len())?;
        for (rank_index, rank) in self.ranks.iter_mut().enumerate() {
            rank.execute(
                &self.stream,
                self.spec,
                &self.selected_preparation,
                &self.bank,
                input,
                selected,
                rank_index,
                0,
                output,
            )?;
        }
        Ok(())
    }

    /// Enqueues projections where every selected expert consumes its own row.
    pub fn execute_ranked(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let input_elements = self
            .ranks
            .len()
            .checked_mul(self.spec.input_features)
            .ok_or(Error::InvalidNvFp4("ranked input size overflow"))?;
        require_len("ranked NVFP4 input", input_elements, input.len())?;
        require_len("ranked NVFP4 indices", self.ranks.len(), selected.len())?;
        require_len("ranked NVFP4 output", self.output_elements()?, output.len())?;
        for (rank_index, rank) in self.ranks.iter_mut().enumerate() {
            rank.execute(
                &self.stream,
                self.spec,
                &self.selected_preparation,
                &self.bank,
                input,
                selected,
                rank_index,
                rank_index * self.spec.input_features,
                output,
            )?;
        }
        Ok(())
    }

    #[must_use]
    pub const fn selected_count(&self) -> usize {
        self.ranks.len()
    }

    #[must_use]
    pub const fn input_features(&self) -> usize {
        self.spec.input_features
    }

    #[must_use]
    pub const fn output_features(&self) -> usize {
        self.spec.output_features
    }

    pub fn output_elements(&self) -> Result<usize> {
        self.ranks
            .len()
            .checked_mul(self.spec.output_features)
            .ok_or(Error::InvalidNvFp4("selected output size overflow"))
    }
}

impl RankWorkspace {
    fn new(backend: &CudaBackend, spec: NvFp4Spec) -> Result<Self> {
        let allocate_u8 =
            |elements| backend.inner.pool.allocate::<u8>(&backend.inner.stream, elements);
        let weight_scales = backend
            .inner
            .pool
            .allocate_zeroed::<u8>(&backend.inner.stream, spec.scale_elements()?)?;
        let input_scales = backend.inner.pool.allocate_zeroed::<u8>(
            &backend.inner.stream,
            scale_elements(1, spec.input_features)?,
        )?;
        let plan = BlockScaledFp4Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            BlockScaledFp4Spec::new(1, spec.output_features, spec.input_features)?,
        )?;
        Ok(Self {
            plan,
            weight: allocate_u8(spec.elements()? / 2)?,
            weight_scales,
            input_global: backend.inner.pool.allocate::<f32>(&backend.inner.stream, 1)?,
            weight_global: backend.inner.pool.allocate::<f32>(&backend.inner.stream, 1)?,
            input: allocate_u8(spec.input_features / 2)?,
            input_scales,
            raw_output: backend
                .inner
                .pool
                .allocate::<bf16>(&backend.inner.stream, spec.output_features)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        stream: &Stream,
        spec: NvFp4Spec,
        selected_preparation: &NvFp4SelectedWeightPreparation,
        bank: &NvFp4ExpertBank,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        rank: usize,
        input_offset: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        selected_preparation.execute(
            stream,
            spec,
            bank.config.experts,
            &mut NvFp4SelectedWeightLaunch {
                source_weight: &bank.weight,
                source_scales: &bank.scales,
                source_input_scales: &bank.input_scales,
                source_weight_scales: &bank.global_scales,
                selected,
                rank,
                weight: &mut self.weight,
                scales: &mut self.weight_scales,
                input_scale: &mut self.input_global,
                weight_scale: &mut self.weight_global,
            },
        )?;
        selected_preparation.quantize(
            stream,
            input,
            input_offset,
            spec.input_features,
            &self.input_global,
            &mut self.input,
            &mut self.input_scales,
        )?;
        self.plan.execute(
            stream,
            &self.input,
            &self.input_scales,
            &self.weight,
            &self.weight_scales,
            &mut self.raw_output,
            1.0,
        )?;
        selected_preparation.scale(
            stream,
            &self.raw_output,
            &self.input_global,
            &self.weight_global,
            output,
            rank * spec.output_features,
        )
    }
}
