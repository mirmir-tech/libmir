use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file, f16,
};

use crate::{Error, Result};

cuda_export!(EmbeddingNormKernel = "libmir_cuda_encoder_embedding_norm_f16"(
    ids: &DeviceBuffer<u32>, words: &DeviceBuffer<f16>, types: &DeviceBuffer<f16>,
    weight: &DeviceBuffer<f16>, bias: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f16>,
    rows: u32, columns: u32, epsilon: f32,
));
cuda_export!(BiasKernel = "libmir_cuda_encoder_bias_f16"(
    input: &DeviceBuffer<f16>, bias: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f16>,
    rows: u32, columns: u32,
));
cuda_export!(ResidualNormKernel = "libmir_cuda_encoder_residual_norm_f16"(
    left: &DeviceBuffer<f16>, right: &DeviceBuffer<f16>, weight: &DeviceBuffer<f16>,
    bias: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f16>, rows: u32, columns: u32,
    epsilon: f32,
));
cuda_export!(GatedKernel = "libmir_cuda_encoder_gated_gelu_f16"(
    input: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f16>, rows: u32, columns: u32,
));
cuda_export!(TanhBiasKernel = "libmir_cuda_encoder_tanh_bias_f16"(
    input: &DeviceBuffer<f16>, bias: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f16>,
    elements: u32,
));

#[derive(Clone, Copy, Debug)]
pub struct EncoderElementwiseSpec {
    pub rows: usize,
    pub columns: usize,
    pub epsilon: f32,
}

#[derive(Clone, Debug)]
pub struct EncoderElementwiseF16 {
    embedding_norm: TypedKernel<EmbeddingNormKernel>,
    bias: TypedKernel<BiasKernel>,
    residual_norm: TypedKernel<ResidualNormKernel>,
    gated: TypedKernel<GatedKernel>,
    tanh_bias: TypedKernel<TanhBiasKernel>,
    spec: EncoderElementwiseSpec,
}

impl EncoderElementwiseF16 {
    pub fn compile(compiler: &Compiler, spec: EncoderElementwiseSpec) -> Result<Self> {
        if spec.rows == 0 || spec.columns == 0 || !spec.epsilon.is_finite() || spec.epsilon < 0.0 {
            return Err(Error::InvalidDecoderKernel("invalid encoder elementwise geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/encoder/elementwise_f16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            embedding_norm: module.kernel()?,
            bias: module.kernel()?,
            residual_norm: module.kernel()?,
            gated: module.kernel()?,
            tanh_bias: module.kernel()?,
            spec,
        })
    }

    #[expect(clippy::too_many_arguments, reason = "typed CUDA kernel boundary")]
    pub fn embedding_norm(
        &self,
        stream: &Stream,
        ids: &DeviceBuffer<u32>,
        words: &DeviceBuffer<f16>,
        types: &DeviceBuffer<f16>,
        weight: &DeviceBuffer<f16>,
        bias: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
    ) -> Result<()> {
        Self::require(output, self.spec.rows * self.spec.columns)?;
        Ok(self.embedding_norm.launch(
            stream,
            row_launch(self.spec.rows)?,
            (
                ids,
                words,
                types,
                weight,
                bias,
                output,
                u32::try_from(self.spec.rows)?,
                u32::try_from(self.spec.columns)?,
                self.spec.epsilon,
            ),
        )?)
    }

    pub fn add_bias(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f16>,
        bias: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
    ) -> Result<()> {
        Self::require(input, self.spec.rows * self.spec.columns)?;
        Self::require(output, input.len())?;
        Ok(self.bias.launch(
            stream,
            element_launch(input.len())?,
            (
                input,
                bias,
                output,
                u32::try_from(self.spec.rows)?,
                u32::try_from(self.spec.columns)?,
            ),
        )?)
    }

    pub fn residual_norm(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<f16>,
        right: &DeviceBuffer<f16>,
        weight: &DeviceBuffer<f16>,
        bias: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
    ) -> Result<()> {
        Self::require(left, self.spec.rows * self.spec.columns)?;
        Self::require(right, left.len())?;
        Self::require(output, left.len())?;
        Ok(self.residual_norm.launch(
            stream,
            row_launch(self.spec.rows)?,
            (
                left,
                right,
                weight,
                bias,
                output,
                u32::try_from(self.spec.rows)?,
                u32::try_from(self.spec.columns)?,
                self.spec.epsilon,
            ),
        )?)
    }

    pub fn gated_gelu(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
    ) -> Result<()> {
        Self::require(input, self.spec.rows * self.spec.columns * 2)?;
        Self::require(output, self.spec.rows * self.spec.columns)?;
        Ok(self.gated.launch(
            stream,
            element_launch(output.len())?,
            (input, output, u32::try_from(self.spec.rows)?, u32::try_from(self.spec.columns)?),
        )?)
    }

    pub fn tanh_bias(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f16>,
        bias: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
    ) -> Result<()> {
        Self::require(input, self.spec.columns)?;
        Self::require(output, self.spec.columns)?;
        Ok(self.tanh_bias.launch(
            stream,
            element_launch(self.spec.columns)?,
            (input, bias, output, u32::try_from(self.spec.columns)?),
        )?)
    }

    fn require<T: mircuda::DeviceElement>(buffer: &DeviceBuffer<T>, expected: usize) -> Result<()> {
        if buffer.len() == expected {
            Ok(())
        } else {
            Err(Error::InvalidDecoderKernel("encoder elementwise buffer geometry differs"))
        }
    }
}

fn row_launch(rows: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (u32::try_from(rows)?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
fn element_launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig::for_elements(elements, 256)?)
}
