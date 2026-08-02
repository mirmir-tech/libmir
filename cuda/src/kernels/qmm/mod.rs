mod kernel;

use kernel::{
    AffineQmmFallbackKernel, AffineQmmInt4Kernel, AffineQmmInt8Kernel, AffineQmmScalarInt4Kernel,
    AffineQmmScalarInt8Kernel,
};
use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16,
    cuda_kernel_file, cuda_kernel_files,
};

use super::{
    affine::{AffineGemvSpec, compile_options},
    geometry::{indexed, narrow, product, require},
};
use crate::{Error, Result};

#[cfg(all(test, target_os = "linux"))]
mod tests;

/// Fixed token and matrix geometry for affine quantized prefill.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AffineQmmSpec {
    pub matrix: AffineGemvSpec,
    pub tokens: usize,
}

impl AffineQmmSpec {
    pub const fn new(matrix: AffineGemvSpec, tokens: usize) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::InvalidQuantizedGemv("token count must be non-zero"));
        }
        if matches!(matrix.bits, 4 | 8)
            && (!matrix.input_features.is_multiple_of(16) || !matrix.group_size.is_multiple_of(16))
        {
            return Err(Error::InvalidQuantizedGemv(
                "tensor-core prefill requires input and group alignment to 16",
            ));
        }
        Ok(Self { matrix, tokens })
    }
}

/// Device buffers and matrix selection for one quantized prefill launch.
pub struct AffineQmmLaunch<'a> {
    pub input: &'a DeviceBuffer<bf16>,
    pub weight: &'a DeviceBuffer<u32>,
    pub scales: &'a DeviceBuffer<bf16>,
    pub biases: &'a DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
    pub matrix_index: usize,
}

/// Tiled BF16-input affine Int4/Int8 quantized matrix multiplication.
#[derive(Clone, Debug)]
pub struct AffineQuantizedQmm {
    kernel: QmmKernel,
    spec: AffineQmmSpec,
}

#[derive(Clone, Debug)]
enum QmmKernel {
    ScalarInt4(TypedKernel<AffineQmmScalarInt4Kernel>),
    ScalarInt8(TypedKernel<AffineQmmScalarInt8Kernel>),
    Int4(TypedKernel<AffineQmmInt4Kernel>),
    Int8(TypedKernel<AffineQmmInt8Kernel>),
    Fallback(TypedKernel<AffineQmmFallbackKernel>),
}

impl AffineQuantizedQmm {
    pub fn compile(compiler: &Compiler, spec: AffineQmmSpec) -> Result<Self> {
        let scalar = spec.tokens < 32 || !matches!(spec.matrix.bits, 4 | 8);
        let source = if scalar {
            cuda_kernel_files!(
                "affine_qmm_scalar_bf16.cu";
                "../../../kernels/affine_packed.cuh",
                "../../../kernels/affine_qmm_scalar_bf16.cu",
            )
        } else {
            cuda_kernel_file!("../../../kernels/affine_qmm_bf16.cu")
        };
        let options = if scalar {
            compile_options(spec.matrix.bits, true)
        } else {
            CompileOptions { fast_math: true, ..Default::default() }
        };
        let module = compiler.compile(source, &options)?;
        let kernel = match (scalar, spec.matrix.bits) {
            (true, 4) => QmmKernel::ScalarInt4(module.kernel()?),
            (true, 8) => QmmKernel::ScalarInt8(module.kernel()?),
            (false, 4) => QmmKernel::Int4(module.kernel()?),
            (false, 8) => QmmKernel::Int8(module.kernel()?),
            (true, 2 | 3 | 5 | 6) => QmmKernel::Fallback(module.kernel()?),
            _ => return Err(Error::InvalidQuantizedGemv("unsupported weight precision")),
        };
        Ok(Self { kernel, spec })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut AffineQmmLaunch<'_>) -> Result<()> {
        self.validate(launch)?;
        let matrix = self.spec.matrix;
        let config = if self.scalar() {
            LaunchConfig {
                grid: (
                    narrow(matrix.output_features.div_ceil(8))?,
                    narrow(self.spec.tokens.div_ceil(8))?,
                    1,
                ),
                block: (32, 8, 1),
                shared_memory_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid: (
                    narrow(matrix.output_features.div_ceil(16))?,
                    narrow(self.spec.tokens.div_ceil(64))?,
                    1,
                ),
                block: (32, 4, 1),
                shared_memory_bytes: 0,
            }
        };
        let dimensions = (
            narrow(self.spec.tokens)?,
            narrow(matrix.input_features)?,
            narrow(matrix.output_features)?,
            narrow(matrix.group_size)?,
            narrow(launch.matrix_index)?,
        );
        match &self.kernel {
            QmmKernel::ScalarInt4(kernel) => {
                launch_kernel(kernel, stream, config, launch, dimensions)
            },
            QmmKernel::ScalarInt8(kernel) => {
                launch_kernel(kernel, stream, config, launch, dimensions)
            },
            QmmKernel::Int4(kernel) => launch_kernel(kernel, stream, config, launch, dimensions),
            QmmKernel::Int8(kernel) => launch_kernel(kernel, stream, config, launch, dimensions),
            QmmKernel::Fallback(kernel) => {
                launch_kernel(kernel, stream, config, launch, dimensions)
            },
        }
    }

    const fn scalar(&self) -> bool {
        matches!(
            self.kernel,
            QmmKernel::ScalarInt4(_) | QmmKernel::ScalarInt8(_) | QmmKernel::Fallback(_)
        )
    }

    #[must_use]
    pub const fn spec(&self) -> AffineQmmSpec {
        self.spec
    }

    fn validate(&self, launch: &AffineQmmLaunch<'_>) -> Result<()> {
        let matrix = self.spec.matrix;
        let layout = matrix.layout()?;
        require(
            "prefill input",
            product(self.spec.tokens, matrix.input_features)?,
            launch.input.len(),
        )?;
        require(
            "weight",
            indexed(layout.packed_per_matrix, launch.matrix_index)?,
            launch.weight.len(),
        )?;
        let grouped = indexed(layout.groups_per_matrix, launch.matrix_index)?;
        require("scales", grouped, launch.scales.len())?;
        require("biases", grouped, launch.biases.len())?;
        require(
            "prefill output",
            product(self.spec.tokens, matrix.output_features)?,
            launch.output.len(),
        )
    }
}

type QmmDimensions = (u32, u32, u32, u32, u32);

fn launch_kernel<S>(
    kernel: &TypedKernel<S>,
    stream: &Stream,
    config: LaunchConfig,
    launch: &mut AffineQmmLaunch<'_>,
    dimensions: QmmDimensions,
) -> Result<()>
where
    for<'a> S: mircuda::KernelSignature<
            Arguments<'a> = (
                &'a DeviceBuffer<bf16>,
                &'a DeviceBuffer<u32>,
                &'a DeviceBuffer<bf16>,
                &'a DeviceBuffer<bf16>,
                &'a mut DeviceBuffer<bf16>,
                u32,
                u32,
                u32,
                u32,
                u32,
            ),
        >,
{
    Ok(kernel.launch(
        stream,
        config,
        (
            launch.input,
            launch.weight,
            launch.scales,
            launch.biases,
            &mut *launch.output,
            dimensions.0,
            dimensions.1,
            dimensions.2,
            dimensions.3,
            dimensions.4,
        ),
    )?)
}
