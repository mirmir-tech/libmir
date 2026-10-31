//! Rows a Mamba-2 layer uses only while it executes: the fused input
//! projection, the mixer output, and the scan intermediates. Layers run one
//! after another, so every Mamba-2 layer of one shape shares a single set.

use mircuda::{DeviceBuffer, bf16};

use crate::{
    CudaBackend, Error, Mamba2Workspace, Result, backend::scratch_pool::SharedScratch,
    kernels::Mamba2Geometry,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct MixerScratchKey {
    tokens: usize,
    projection: usize,
    channels: usize,
    inner: usize,
}

#[derive(Debug)]
pub(in crate::backend) struct MixerScratch {
    pub(super) projection: DeviceBuffer<bf16>,
    pub(super) mixed: DeviceBuffer<bf16>,
    pub(super) workspace: Mamba2Workspace,
}

impl CudaBackend {
    pub(super) fn mamba2_mixer_scratch(
        &self,
        geometry: Mamba2Geometry,
        tokens: usize,
    ) -> Result<SharedScratch<MixerScratch>> {
        let (projection, inner) = (geometry.projection_width()?, geometry.inner()?);
        let key = MixerScratchKey {
            tokens,
            projection,
            channels: geometry.convolution_channels()?,
            inner,
        };
        let rows = |width: usize| {
            tokens
                .checked_mul(width)
                .ok_or(Error::InvalidDecoderKernel("Mamba-2 layer size overflow"))
        };
        self.inner.mamba2_scratch.acquire(key, || {
            let (stream, pool) = (&self.inner.stream, &self.inner.pool);
            Ok(MixerScratch {
                projection: pool.allocate(stream, rows(projection)?)?,
                mixed: pool.allocate(stream, rows(inner)?)?,
                workspace: self.prepare_mamba2_workspace(geometry, tokens)?,
            })
        })
    }
}
