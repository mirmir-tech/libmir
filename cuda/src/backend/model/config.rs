use crate::{Error, Result};

pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 1_024;

/// Explicit allocation policy for one CUDA model session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaModelSessionConfig {
    /// Maximum prompt tokens uploaded in one pinned-host transfer.
    pub prefill_chunk_tokens: usize,
}

impl Default for CudaModelSessionConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
        }
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

#[cfg(test)]
mod tests {
    use super::CudaModelSessionConfig;

    #[test]
    fn default_prefill_chunk_matches_the_tuned_gb10_shape() {
        assert_eq!(CudaModelSessionConfig::default().prefill_chunk_tokens, 1_024);
    }
}
