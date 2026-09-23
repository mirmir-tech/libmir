//! Batched convolution and recurrence kernels compiled per row count.

use std::collections::hash_map::Entry;

use super::{CudaGatedDeltaBatchState, channels};
use crate::{
    Error, Result,
    kernels::{
        GatedDeltaBatchConvolution, GatedDeltaBatchConvolutionSpec, GatedDeltaBatchRecurrence,
        GatedDeltaBatchSpec,
    },
};

/// Kernels of one row count; the buffers serve every count up to capacity.
#[derive(Debug)]
pub(super) struct BatchKernels {
    pub(super) convolution: GatedDeltaBatchConvolution,
    pub(super) recurrence: GatedDeltaBatchRecurrence,
}

impl CudaGatedDeltaBatchState {
    pub(super) fn kernels(&mut self, rows: usize) -> Result<&mut BatchKernels> {
        let backend = &self.backend;
        let config = self.config;
        let tokens = self.tokens;
        if let Entry::Vacant(entry) = self.kernels.entry(rows) {
            entry.insert(BatchKernels {
                convolution: GatedDeltaBatchConvolution::compile(
                    &backend.inner.compiler,
                    GatedDeltaBatchConvolutionSpec {
                        rows,
                        tokens,
                        channels: channels(config)?,
                        kernel_size: config.convolution_kernel_size,
                    },
                )?,
                recurrence: GatedDeltaBatchRecurrence::compile(
                    &backend.inner.compiler,
                    GatedDeltaBatchSpec {
                        rows,
                        tokens,
                        key_heads: config.key_heads,
                        value_heads: config.value_heads,
                        key_dim: config.key_dim,
                        value_dim: config.value_dim,
                    },
                )?,
            });
        }
        self.kernels
            .get_mut(&rows)
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed kernels are missing"))
    }
}
