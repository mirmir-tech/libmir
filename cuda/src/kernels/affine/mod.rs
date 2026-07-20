mod embedding;

pub use embedding::{AffineEmbedding, AffineEmbeddingSpec};
use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{Layout, indexed, narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    AffineGemvInt4Kernel = "libmir_cuda_affine_gemv_bf16_int4"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<bf16>,
        biases: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_features: u32,
        output_features: u32,
        group_size: u32,
        matrix_index: u32,
    )
);

cuda_export!(
    AffineGemvInt8Kernel = "libmir_cuda_affine_gemv_bf16_int8"(
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<bf16>,
        biases: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        input_features: u32,
        output_features: u32,
        group_size: u32,
        matrix_index: u32,
    )
);

/// Geometry of one affine grouped quantized matrix.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AffineGemvSpec {
    /// Number of input features before packing.
    pub input_features: usize,
    /// Number of output rows.
    pub output_features: usize,
    /// Number of adjacent input values sharing scale and bias.
    pub group_size: usize,
    /// Packed weight precision. Four and eight bits are supported.
    pub bits: usize,
}

impl AffineGemvSpec {
    /// Creates and validates a grouped affine GEMV shape.
    pub fn new(
        input_features: usize,
        output_features: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<Self> {
        let spec = Self {
            input_features,
            output_features,
            group_size,
            bits,
        };
        let _layout = spec.layout()?;
        Ok(spec)
    }

    pub(super) fn layout(self) -> Result<Layout> {
        if self.input_features == 0 || self.output_features == 0 || self.group_size == 0 {
            return Err(Error::InvalidQuantizedGemv("dimensions must be non-zero"));
        }
        if !matches!(self.bits, 4 | 8) {
            return Err(Error::InvalidQuantizedGemv(
                "only four and eight bit weights are supported",
            ));
        }
        if !self.input_features.is_multiple_of(self.group_size) {
            return Err(Error::InvalidQuantizedGemv("input features must divide into groups"));
        }
        let values_per_word = 32 / self.bits;
        let values_per_thread = if self.bits == 4 {
            16
        } else {
            8
        };
        if !self.group_size.is_multiple_of(values_per_thread) {
            return Err(Error::InvalidQuantizedGemv(
                "group size must align with one warp-thread input tile",
            ));
        }
        if !self.input_features.is_multiple_of(values_per_word) {
            return Err(Error::InvalidQuantizedGemv("packed rows must end on a U32 boundary"));
        }
        Ok(Layout {
            packed_per_matrix: product(
                self.output_features,
                self.input_features / values_per_word,
            )?,
            groups_per_matrix: product(
                self.output_features,
                self.input_features / self.group_size,
            )?,
        })
    }
}

/// Compiled BF16-input affine grouped quantized GEMV operation.
#[derive(Clone, Debug)]
pub struct AffineQuantizedGemv {
    kernel: AffineKernel,
    spec: AffineGemvSpec,
}

#[derive(Clone, Debug)]
enum AffineKernel {
    Int4(TypedKernel<AffineGemvInt4Kernel>),
    Int8(TypedKernel<AffineGemvInt8Kernel>),
}

/// Buffers and matrix selection for one quantized GEMV launch.
pub struct AffineGemvLaunch<'a> {
    /// One BF16 input vector.
    pub input: &'a DeviceBuffer<bf16>,
    /// Packed affine weights, optionally containing multiple matrices.
    pub weight: &'a DeviceBuffer<u32>,
    /// Per-output, per-group BF16 scales.
    pub scales: &'a DeviceBuffer<bf16>,
    /// Per-output, per-group BF16 affine biases.
    pub biases: &'a DeviceBuffer<bf16>,
    /// One BF16 output vector.
    pub output: &'a mut DeviceBuffer<bf16>,
    /// Leading matrix index for expert tensors; zero for a 2D matrix.
    pub matrix_index: usize,
}

impl AffineQuantizedGemv {
    /// Compiles or retrieves the kernel module and fixes its matrix geometry.
    pub fn compile(compiler: &Compiler, spec: AffineGemvSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/affine_gemv_bf16.cu");
        let module =
            compiler.compile(source, &CompileOptions { fast_math: true, ..Default::default() })?;
        let kernel = match spec.bits {
            4 => AffineKernel::Int4(module.kernel()?),
            8 => AffineKernel::Int8(module.kernel()?),
            _ => return Err(Error::InvalidQuantizedGemv("unsupported weight precision")),
        };
        Ok(Self { kernel, spec })
    }

    /// Enqueues `output = input x dequantized(weight[matrix_index])^T`.
    pub fn execute(&self, stream: &Stream, launch: &mut AffineGemvLaunch<'_>) -> Result<()> {
        let layout = self.spec.layout()?;
        require("input", self.spec.input_features, launch.input.len())?;
        require(
            "weight",
            indexed(layout.packed_per_matrix, launch.matrix_index)?,
            launch.weight.len(),
        )?;
        let grouped = indexed(layout.groups_per_matrix, launch.matrix_index)?;
        require("scales", grouped, launch.scales.len())?;
        require("biases", grouped, launch.biases.len())?;
        require("output", self.spec.output_features, launch.output.len())?;
        let config = LaunchConfig {
            grid: (narrow(self.spec.output_features.div_ceil(8))?, 1, 1),
            block: (32, 8, 1),
            shared_memory_bytes: 0,
        };
        let dimensions = (
            narrow(self.spec.input_features)?,
            narrow(self.spec.output_features)?,
            narrow(self.spec.group_size)?,
            narrow(launch.matrix_index)?,
        );
        Ok(match &self.kernel {
            AffineKernel::Int4(kernel) => kernel.launch(
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
                ),
            ),
            AffineKernel::Int8(kernel) => kernel.launch(
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
                ),
            ),
        }?)
    }

    /// Returns the fixed matrix geometry.
    #[must_use]
    pub const fn spec(&self) -> AffineGemvSpec {
        self.spec
    }
}
