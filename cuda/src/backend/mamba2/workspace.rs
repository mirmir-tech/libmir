//! Chunk-sized intermediates of one Mamba-2 advance. They hold nothing
//! between calls, so one workspace serves every layer and sequence of a plan
//! instead of each state keeping its own prefill-sized copy.

use mircuda::{DeviceBuffer, bf16};

use super::CudaBackend;
use crate::{Error, Result, kernels::Mamba2Geometry};

#[derive(Debug)]
pub struct Mamba2Workspace {
    pub(super) convolved: DeviceBuffer<bf16>,
    pub(super) scanned: DeviceBuffer<bf16>,
    tokens: usize,
}

impl Mamba2Workspace {
    /// Rows this workspace covers.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}

impl CudaBackend {
    pub fn prepare_mamba2_workspace(
        &self,
        geometry: Mamba2Geometry,
        tokens: usize,
    ) -> Result<Mamba2Workspace> {
        let rows = |width: usize| {
            tokens
                .checked_mul(width)
                .ok_or(Error::InvalidDecoderKernel("Mamba-2 workspace overflow"))
        };
        let (stream, pool) = (&self.inner.stream, &self.inner.pool);
        Ok(Mamba2Workspace {
            convolved: pool.allocate(stream, rows(geometry.convolution_channels()?)?)?,
            scanned: pool.allocate(stream, rows(geometry.inner()?)?)?,
            tokens,
        })
    }
}
