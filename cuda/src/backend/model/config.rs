use crate::{Error, Result};

/// Explicit allocation policy for one CUDA model session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaModelSessionConfig {
    /// Maximum prompt tokens uploaded in one pinned-host transfer.
    pub prefill_chunk_tokens: usize,
}

impl Default for CudaModelSessionConfig {
    fn default() -> Self {
        Self { prefill_chunk_tokens: 256 }
    }
}

impl CudaModelSessionConfig {
    pub(super) fn validate(self) -> Result<Self> {
        if self.prefill_chunk_tokens == 0 {
            Err(Error::InvalidDecoderKernel("CUDA prefill chunk cannot be empty"))
        } else {
            Ok(self)
        }
    }
}
