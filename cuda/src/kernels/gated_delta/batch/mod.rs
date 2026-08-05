use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(
    BatchConvolutionKernel = "libmir_cuda_gated_delta_batch_convolution_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        history: &DeviceBuffer<bf16>, next_history: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, channels: u32, kernel_size: u32, tokens: u32,
        input_stride: u32, input_offset: u32,
    )
);

mod recurrence;
pub use recurrence::{GatedDeltaBatchRecurrence, GatedDeltaBatchSpec};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaBatchConvolutionSpec {
    pub rows: usize,
    pub tokens: usize,
    pub channels: usize,
    pub kernel_size: usize,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaBatchConvolution {
    kernel: TypedKernel<BatchConvolutionKernel>,
    spec: GatedDeltaBatchConvolutionSpec,
}

impl GatedDeltaBatchConvolution {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaBatchConvolutionSpec) -> Result<Self> {
        if spec.rows == 0 || spec.tokens == 0 || spec.channels == 0 || spec.kernel_size < 2 {
            return Err(Error::InvalidDecoderKernel("invalid batched Gated Delta convolution"));
        }
        let source = cuda_kernel_file!("../../../../kernels/gated_delta_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        history: &DeviceBuffer<bf16>,
        next_history: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_strided(
            stream,
            input,
            weight,
            history,
            next_history,
            output,
            self.spec.channels,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_strided(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        history: &DeviceBuffer<bf16>,
        next_history: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<()> {
        let values = product(product(self.spec.rows, self.spec.tokens)?, self.spec.channels)?;
        let history_values =
            product(product(self.spec.rows, self.spec.channels)?, self.spec.kernel_size - 1)?;
        let packed_tokens = product(self.spec.rows, self.spec.tokens)?;
        validate_input(input, packed_tokens, self.spec.channels, input_stride, input_offset)?;
        require(
            "batched Gated Delta weight",
            product(self.spec.channels, self.spec.kernel_size)?,
            weight.len(),
        )?;
        require("batched Gated Delta history", history_values, history.len())?;
        require("batched Gated Delta next history", history_values, next_history.len())?;
        require("batched Gated Delta output", values, output.len())?;
        Ok(self.kernel.launch(
            stream,
            linear_launch(values)?,
            (
                input,
                weight,
                history,
                next_history,
                output,
                narrow(self.spec.rows)?,
                narrow(self.spec.channels)?,
                narrow(self.spec.kernel_size)?,
                narrow(self.spec.tokens)?,
                narrow(input_stride)?,
                narrow(input_offset)?,
            ),
        )?)
    }

    pub fn execute_in_place(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        history: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        // The kernel reads every channel's old history before shifting that same
        // channel toward lower indices, so aliasing both history arguments is safe.
        let source = history.clone();
        self.execute(stream, input, weight, &source, history, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_in_place_strided(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        history: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_stride: usize,
        input_offset: usize,
    ) -> Result<()> {
        let source = history.clone();
        self.execute_strided(
            stream, input, weight, &source, history, output, input_stride, input_offset,
        )
    }
}

fn validate_input(
    input: &DeviceBuffer<bf16>,
    tokens: usize,
    channels: usize,
    stride: usize,
    offset: usize,
) -> Result<()> {
    let row_end = offset
        .checked_add(channels)
        .filter(|end| *end <= stride)
        .ok_or(Error::InvalidDecoderKernel("invalid batched Gated Delta input stride"))?;
    let required = product(tokens.saturating_sub(1), stride)?
        .checked_add(row_end)
        .ok_or(Error::InvalidDecoderKernel("batched Gated Delta input stride overflow"))?;
    if input.len() < required {
        return Err(Error::InvalidDecoderKernel("strided batched Gated Delta input is too small"));
    }
    Ok(())
}

fn linear_launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
