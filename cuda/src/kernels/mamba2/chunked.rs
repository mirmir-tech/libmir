//! Chunked structured state-space duality (SSD) prefill. Equivalent to the
//! sequential scan up to floating-point association, but each chunk of
//! `MAMBA2_CHUNK` tokens is processed with tensor-core matrix products.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{Mamba2Geometry, Mamba2Kernels, Mamba2Launch};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product},
};

/// Tokens per SSD chunk; must match `LIBMIR_MAMBA2_CHUNK` in the kernel.
pub const MAMBA2_CHUNK: usize = 64;

/// BF16 row padding of shared-memory tiles; must match `LIBMIR_MAMBA2_PAD`.
const PAD: usize = 8;

/// Threads of the chunked scan; must match `LIBMIR_MAMBA2_SCAN_THREADS`.
const SCAN_THREADS: u32 = 512;

cuda_export!(
    ChunkScanKernel = "libmir_cuda_mamba2_chunk_scan_bf16"(
        convolved: &DeviceBuffer<bf16>, projection: &DeviceBuffer<bf16>,
        a_log: &DeviceBuffer<bf16>, skip: &DeviceBuffer<bf16>, dt_bias: &DeviceBuffer<bf16>,
        state: &mut DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, tokens: u32, heads: u32,
        head_dim: u32, state_size: u32, groups: u32, projection_stride: u32, dt_offset: u32,
        dt_minimum: f32, dt_maximum: f32,
    )
);

/// The chunked scan kernel, present only when the geometry fits it.
#[derive(Clone, Debug)]
pub(super) struct ChunkedKernels {
    scan: Option<TypedKernel<ChunkScanKernel>>,
}

impl ChunkedKernels {
    pub(super) fn compile(compiler: &Compiler, geometry: Mamba2Geometry) -> Result<Self> {
        if !geometry.supports_chunked_scan() {
            return Ok(Self { scan: None });
        }
        let source = cuda_kernel_file!("../../../kernels/mamba2_chunked_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        let scan: TypedKernel<ChunkScanKernel> = module.kernel()?;
        scan.set_max_dynamic_shared_memory_bytes(narrow(geometry.chunked_scan_bytes()?)?)?;
        Ok(Self { scan: Some(scan) })
    }
}

impl Mamba2Geometry {
    /// Whether the chunked scan fits this geometry: at most 16 output tiles
    /// per chunk, operands loadable as 16-byte vectors, and room for FP32
    /// outputs in the BF16 `C` region of shared memory.
    #[must_use]
    pub fn supports_chunked_scan(&self) -> bool {
        self.head_dim <= 64
            && self.state_size <= 128
            && 2 * self.head_dim <= self.state_size
            && self.head_dim.is_multiple_of(16)
            && self.state_size.is_multiple_of(16)
    }

    /// Dynamic shared memory of the chunked scan; see the layout comment on
    /// `libmir_cuda_mamba2_chunk_scan_bf16`.
    fn chunked_scan_bytes(&self) -> Result<usize> {
        let wide = self.state_size + PAD;
        let halves = product(MAMBA2_CHUNK, wide)?
            + product(MAMBA2_CHUNK.max(self.head_dim), wide)?
            + product(MAMBA2_CHUNK, self.head_dim + PAD)?;
        let floats = product(MAMBA2_CHUNK, MAMBA2_CHUNK + 4)?
            + product(self.head_dim, self.state_size + 4)?
            + 2 * MAMBA2_CHUNK;
        Ok(product(halves, 2)? + product(floats, 4)?)
    }
}

impl Mamba2Kernels {
    /// Runs convolution, the per-head chunked scan, and gated norm. One block
    /// per head walks every chunk in order and keeps the FP32 state on chip.
    pub fn execute_chunked(&self, stream: &Stream, launch: &mut Mamba2Launch<'_>) -> Result<()> {
        self.validate(launch)?;
        let geometry = self.geometry;
        let kernel = self.chunked.scan.as_ref().ok_or(Error::InvalidDecoderKernel(
            "geometry does not support the chunked Mamba-2 scan",
        ))?;
        self.convolve(stream, launch)?;
        let parameters = launch.parameters;
        kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(geometry.heads)?, 1, 1),
                block: (SCAN_THREADS, 1, 1),
                shared_memory_bytes: narrow(geometry.chunked_scan_bytes()?)?,
            },
            (
                &*launch.convolved,
                launch.projection,
                parameters.a_log,
                parameters.skip,
                parameters.dt_bias,
                &mut *launch.state,
                &mut *launch.scanned,
                narrow(launch.tokens)?,
                narrow(geometry.heads)?,
                narrow(geometry.head_dim)?,
                narrow(geometry.state_size)?,
                narrow(geometry.groups)?,
                narrow(geometry.projection_width()?)?,
                narrow(geometry.inner()? + geometry.convolution_channels()?)?,
                geometry.time_step_minimum,
                geometry.time_step_maximum,
            ),
        )?;
        self.normalize(stream, launch)
    }
}
