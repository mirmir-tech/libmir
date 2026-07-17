use mircuda::{DeviceBuffer, Stream, bf16};

use super::CudaBackend;
use crate::{CudaTensor, Error, Result, kernels::Embedding};

/// Fixed-shape BF16 token embedding lookup.
#[derive(Clone, Debug)]
pub struct Bf16Embedding {
    operation: Embedding,
    stream: Stream,
    vocab: usize,
    hidden: usize,
}

impl CudaBackend {
    pub fn prepare_bf16_embedding(
        &self,
        vocab: usize,
        hidden: usize,
        scale: f32,
    ) -> Result<Bf16Embedding> {
        Ok(Bf16Embedding {
            operation: Embedding::compile(&self.inner.compiler, vocab, hidden, scale)?,
            stream: self.inner.stream.clone(),
            vocab,
            hidden,
        })
    }
}

impl Bf16Embedding {
    pub fn execute(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_index: usize,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if weight.shape() != [self.vocab, self.hidden] {
            return Err(Error::InvalidQuantizedTensor {
                name: weight.name().into(),
                expected: vec![self.vocab, self.hidden],
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        self.operation.execute(&self.stream, weight, selected, selected_index, output)
    }

    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if weight.shape() != [self.vocab, self.hidden] {
            return Err(Error::InvalidQuantizedTensor {
                name: weight.name().into(),
                expected: vec![self.vocab, self.hidden],
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        self.operation
            .execute_batch(&self.stream, weight, selected, selected_start, tokens, output)
    }

    pub(crate) fn validate_token(&self, token: u32) -> Result<()> {
        if usize::try_from(token)? < self.vocab {
            Ok(())
        } else {
            Err(Error::InvalidToken { token, vocab: self.vocab })
        }
    }
}
