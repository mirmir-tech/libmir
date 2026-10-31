//! Mamba-2 selective state-space mixer kernels: causal convolution with
//! bias and `SiLU`, a sequential selective scan with FP32 state, and the gated
//! grouped `RMSNorm` applied before the output projection.

use mircuda::{CompileOptions, Compiler, TypedKernel, bf16, cuda_export, cuda_kernel_file};

use super::geometry::product;
use crate::{Error, Result};

mod batch;
mod chunked;
mod launch;

pub use batch::Mamba2DecodeBatch;
pub use chunked::MAMBA2_CHUNK;
pub use launch::{Mamba2Launch, Mamba2Parameters};

/// Rows of one head's state handled by one scan thread block.
pub const MAMBA2_SCAN_ROWS: usize = 16;

cuda_export!(
    ConvolutionKernel = "libmir_cuda_mamba2_convolution_bf16"(
        input: &mircuda::DeviceBuffer<bf16>, weight: &mircuda::DeviceBuffer<bf16>,
        bias: &mircuda::DeviceBuffer<bf16>, history: &mircuda::DeviceBuffer<bf16>,
        output: &mut mircuda::DeviceBuffer<bf16>, tokens: u32, channels: u32,
        kernel_size: u32, input_stride: u32, input_offset: u32, has_bias: u32,
    )
);
cuda_export!(
    HistoryKernel = "libmir_cuda_mamba2_history_bf16"(
        input: &mircuda::DeviceBuffer<bf16>, history: &mircuda::DeviceBuffer<bf16>,
        next_history: &mut mircuda::DeviceBuffer<bf16>, tokens: u32, channels: u32,
        kernel_size: u32, input_stride: u32, input_offset: u32,
    )
);
cuda_export!(
    ScanKernel = "libmir_cuda_mamba2_scan_bf16"(
        convolved: &mircuda::DeviceBuffer<bf16>, projection: &mircuda::DeviceBuffer<bf16>,
        a_log: &mircuda::DeviceBuffer<bf16>, skip: &mircuda::DeviceBuffer<bf16>,
        dt_bias: &mircuda::DeviceBuffer<bf16>, state: &mut mircuda::DeviceBuffer<f32>,
        output: &mut mircuda::DeviceBuffer<bf16>, tokens: u32, heads: u32, head_dim: u32,
        state_size: u32, groups: u32, projection_stride: u32, dt_offset: u32,
        dt_minimum: f32, dt_maximum: f32,
    )
);
cuda_export!(
    GatedNormKernel = "libmir_cuda_mamba2_gated_norm_bf16"(
        values: &mircuda::DeviceBuffer<bf16>, projection: &mircuda::DeviceBuffer<bf16>,
        weight: &mircuda::DeviceBuffer<bf16>, output: &mut mircuda::DeviceBuffer<bf16>,
        tokens: u32, inner: u32, group_size: u32, projection_stride: u32, epsilon: f32,
    )
);

/// Geometry of one Mamba-2 mixer, independent of the token count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba2Geometry {
    pub heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub groups: usize,
    pub convolution_kernel_size: usize,
    pub time_step_minimum: f32,
    /// `f32::INFINITY` when the time step is unbounded.
    pub time_step_maximum: f32,
    pub norm_epsilon: f32,
}

impl Mamba2Geometry {
    pub fn inner(&self) -> Result<usize> {
        product(self.heads, self.head_dim)
    }

    /// Channels through the convolution: `x | B | C`.
    pub fn convolution_channels(&self) -> Result<usize> {
        let groups = product(product(self.groups, self.state_size)?, 2)?;
        self.inner()?
            .checked_add(groups)
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 width overflow"))
    }

    /// Width of the fused `z | xBC | dt` input projection.
    pub fn projection_width(&self) -> Result<usize> {
        self.inner()?
            .checked_add(self.convolution_channels()?)
            .and_then(|width| width.checked_add(self.heads))
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 projection width overflow"))
    }

    pub fn state_elements(&self) -> Result<usize> {
        product(self.inner()?, self.state_size)
    }

    pub fn history_elements(&self) -> Result<usize> {
        product(self.convolution_kernel_size - 1, self.convolution_channels()?)
    }

    fn validate(&self) -> Result<()> {
        let valid = self.heads > 0
            && self.groups > 0
            && self.heads.is_multiple_of(self.groups)
            && self.head_dim.is_multiple_of(MAMBA2_SCAN_ROWS)
            && self.head_dim > 0
            && self.state_size.is_multiple_of(32)
            && (32..=1024).contains(&self.state_size)
            && self.convolution_kernel_size >= 2
            && self.inner()?.is_multiple_of(self.groups)
            && self.time_step_minimum >= 0.0
            && self.time_step_maximum >= self.time_step_minimum
            && self.norm_epsilon > 0.0;
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidDecoderKernel("unsupported Mamba-2 geometry"))
        }
    }
}

/// Compiled Mamba-2 kernels for one geometry.
#[derive(Clone, Debug)]
pub struct Mamba2Kernels {
    convolution: TypedKernel<ConvolutionKernel>,
    history: TypedKernel<HistoryKernel>,
    scan: TypedKernel<ScanKernel>,
    gated_norm: TypedKernel<GatedNormKernel>,
    decode_convolution: TypedKernel<batch::DecodeConvolutionKernel>,
    decode_scan: TypedKernel<batch::DecodeScanKernel>,
    chunked: chunked::ChunkedKernels,
    geometry: Mamba2Geometry,
}

impl Mamba2Kernels {
    pub fn compile(compiler: &Compiler, geometry: Mamba2Geometry) -> Result<Self> {
        geometry.validate()?;
        let source = cuda_kernel_file!("../../../kernels/mamba2_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        let chunked = chunked::ChunkedKernels::compile(compiler, geometry)?;
        Ok(Self {
            convolution: module.kernel()?,
            history: module.kernel()?,
            scan: module.kernel()?,
            gated_norm: module.kernel()?,
            decode_convolution: module.kernel()?,
            decode_scan: module.kernel()?,
            chunked,
            geometry,
        })
    }

    #[must_use]
    pub const fn geometry(&self) -> Mamba2Geometry {
        self.geometry
    }
}
