use mircuda::{DeviceBuffer, bf16};

use super::{CudaGatedDeltaState, channels};
use crate::{
    Result,
    kernels::{GatedDeltaConvolution, GatedDeltaConvolutionSpec},
};

impl CudaGatedDeltaState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn convolve_silu_split_normalize_strided(
        &mut self,
        tokens: usize,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        normalized_query: &mut DeviceBuffer<bf16>,
        normalized_key: &mut DeviceBuffer<bf16>,
        value: &mut DeviceBuffer<bf16>,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<()> {
        self.materialize()?;
        let operation = GatedDeltaConvolution::compile(
            &self.backend.inner.compiler,
            GatedDeltaConvolutionSpec {
                tokens,
                channels: channels(self.config)?,
                kernel_size: self.config.convolution_kernel_size,
            },
        )?;
        operation.execute_split_normalize_128_strided(
            &self.backend.inner.stream,
            input,
            weight,
            &self.convolution,
            &mut self.next_convolution,
            normalized_query,
            normalized_key,
            value,
            self.config.key_heads,
            self.config.value_heads,
            self.config.value_dim,
            input_stride,
            input_offset,
        )?;
        std::mem::swap(&mut self.convolution, &mut self.next_convolution);
        Ok(())
    }

    pub fn convolve_silu(
        &mut self,
        tokens: usize,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let channels = channels(self.config)?;
        self.convolve_silu_strided(tokens, input, weight, output, channels, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn convolve_silu_strided(
        &mut self,
        tokens: usize,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<()> {
        self.materialize()?;
        let operation = GatedDeltaConvolution::compile(
            &self.backend.inner.compiler,
            GatedDeltaConvolutionSpec {
                tokens,
                channels: channels(self.config)?,
                kernel_size: self.config.convolution_kernel_size,
            },
        )?;
        operation.execute_strided(
            &self.backend.inner.stream,
            input,
            weight,
            &self.convolution,
            &mut self.next_convolution,
            output,
            input_stride,
            input_offset,
        )?;
        std::mem::swap(&mut self.convolution, &mut self.next_convolution);
        Ok(())
    }
}
