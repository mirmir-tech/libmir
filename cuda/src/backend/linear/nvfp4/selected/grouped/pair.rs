use mircuda::{DeviceBuffer, bf16};

use super::GroupedNvFp4LinearBf16;
use crate::{CudaBackend, NvFp4ExpertBank, Result};

#[derive(Debug)]
pub(in crate::backend::linear::nvfp4::selected) struct GroupedNvFp4PairBf16 {
    left: GroupedNvFp4LinearBf16,
    right: GroupedNvFp4LinearBf16,
}

impl GroupedNvFp4PairBf16 {
    pub(in crate::backend::linear::nvfp4::selected) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        left: NvFp4ExpertBank,
        right: NvFp4ExpertBank,
    ) -> Result<Self> {
        Ok(Self {
            left: GroupedNvFp4LinearBf16::new(backend, tokens, selected, left)?,
            right: GroupedNvFp4LinearBf16::new(backend, tokens, selected, right)?,
        })
    }

    pub(in crate::backend::linear::nvfp4::selected) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        left_output: &mut DeviceBuffer<bf16>,
        right_output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        GroupedNvFp4LinearBf16::quantize_pair(&mut self.left, &mut self.right, input, selected)?;
        self.left.execute_prepared(selected, left_output)?;
        self.right.execute_prepared(selected, right_output)
    }

    pub(in crate::backend::linear::nvfp4::selected) fn output_elements(&self) -> Result<usize> {
        self.left.output_elements()
    }

    pub(in crate::backend::linear::nvfp4::selected) const fn output_features(&self) -> usize {
        self.left.output_features()
    }
}
