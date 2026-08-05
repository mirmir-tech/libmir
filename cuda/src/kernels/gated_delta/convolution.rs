use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    ConvolutionKernel = "libmir_cuda_gated_delta_convolution_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        history: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        tokens: u32, channels: u32, kernel_size: u32,
        input_stride: u32, input_offset: u32,
    )
);
cuda_export!(
    HistoryKernel = "libmir_cuda_gated_delta_history_bf16"(
        input: &DeviceBuffer<bf16>, history: &DeviceBuffer<bf16>,
        next_history: &mut DeviceBuffer<bf16>, tokens: u32,
        channels: u32, kernel_size: u32, input_stride: u32, input_offset: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaConvolutionSpec {
    pub tokens: usize,
    pub channels: usize,
    pub kernel_size: usize,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaConvolution {
    convolution: TypedKernel<ConvolutionKernel>,
    history: TypedKernel<HistoryKernel>,
    spec: GatedDeltaConvolutionSpec,
}

impl GatedDeltaConvolution {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaConvolutionSpec) -> Result<Self> {
        if spec.tokens == 0 || spec.channels == 0 || spec.kernel_size < 2 {
            return Err(Error::InvalidDecoderKernel("invalid Gated Delta convolution geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/gated_delta_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            convolution: module.kernel()?,
            history: module.kernel()?,
            spec,
        })
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
        let values = product(self.spec.tokens, self.spec.channels)?;
        let history_values = self.history_elements()?;
        validate_input(self.spec, input, input_stride, input_offset)?;
        require(
            "Gated Delta convolution weight",
            product(self.spec.channels, self.spec.kernel_size)?,
            weight.len(),
        )?;
        require("Gated Delta convolution history", history_values, history.len())?;
        require("Gated Delta next convolution history", history_values, next_history.len())?;
        require("Gated Delta convolution output", values, output.len())?;
        let arguments = (
            narrow(self.spec.tokens)?,
            narrow(self.spec.channels)?,
            narrow(self.spec.kernel_size)?,
            narrow(input_stride)?,
            narrow(input_offset)?,
        );
        self.convolution.launch(
            stream,
            launch(values)?,
            (
                input, weight, history, output, arguments.0, arguments.1, arguments.2, arguments.3,
                arguments.4,
            ),
        )?;
        Ok(self.history.launch(
            stream,
            launch(history_values)?,
            (
                input, history, next_history, arguments.0, arguments.1, arguments.2, arguments.3,
                arguments.4,
            ),
        )?)
    }

    pub fn history_elements(&self) -> Result<usize> {
        product(self.spec.kernel_size - 1, self.spec.channels)
    }
}

fn validate_input(
    spec: GatedDeltaConvolutionSpec,
    input: &DeviceBuffer<bf16>,
    stride: usize,
    offset: usize,
) -> Result<()> {
    let row_end = offset
        .checked_add(spec.channels)
        .filter(|end| *end <= stride)
        .ok_or(Error::InvalidDecoderKernel("invalid Gated Delta input stride"))?;
    let required = product(spec.tokens.saturating_sub(1), stride)?
        .checked_add(row_end)
        .ok_or(Error::InvalidDecoderKernel("Gated Delta input stride overflow"))?;
    if input.len() < required {
        return Err(Error::InvalidDecoderKernel("strided Gated Delta input is too small"));
    }
    Ok(())
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
