use mircuda::{DeviceBuffer, bf16};

use crate::{AffineQuantizedWeight, CudaBackend, Error, Result, backend::linear::AffineProjection};

/// Single-token affine Int4/Int8 language-model output projection.
#[derive(Debug)]
pub struct CudaAffineOutputHead {
    projection: AffineProjection,
    hidden_size: usize,
    vocab_size: usize,
}

impl CudaAffineOutputHead {
    pub fn from_weight(
        backend: &CudaBackend,
        hidden_size: usize,
        vocab_size: usize,
        weight: &AffineQuantizedWeight,
    ) -> Result<Self> {
        let config = weight.infer_config(1, hidden_size, vocab_size)?;
        Self::new(backend, hidden_size, vocab_size, config.group_size, config.bits, weight)
    }

    pub fn new(
        backend: &CudaBackend,
        hidden_size: usize,
        vocab_size: usize,
        group_size: usize,
        bits: usize,
        weight: &AffineQuantizedWeight,
    ) -> Result<Self> {
        weight.validate(1, hidden_size, vocab_size, group_size, bits)?;
        Ok(Self {
            projection: AffineProjection::new(
                backend, 1, hidden_size, vocab_size, group_size, bits, weight,
            )?,
            hidden_size,
            vocab_size,
        })
    }

    /// Enqueues full-vocabulary logits without host synchronization.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        logits: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if input.len() != self.hidden_size || logits.len() != self.vocab_size {
            return Err(Error::InvalidDecoderKernel("affine output-head buffer mismatch"));
        }
        self.projection.execute(input, logits)
    }
}
