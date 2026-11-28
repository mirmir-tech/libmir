use mircuda::{DeviceBuffer, bf16};

use crate::{
    AffineQuantizedEmbedding, Bf16Embedding, CudaBackend, DenseRole, Error, GgmlEmbeddingLookup,
    MxFp4EmbeddingLookup, Result,
    backend::linear::{
        CheckpointProjection, CheckpointProjectionWeight, CompressedInt8Embedding,
        packed_integer_embedding,
    },
};

#[derive(Debug)]
pub(super) enum SharedRoutedEmbedding {
    Affine {
        operation: AffineQuantizedEmbedding,
        weight: crate::AffineQuantizedWeight,
        vocab: usize,
    },
    Dense {
        operation: Bf16Embedding,
        weight: crate::CudaTensor,
        vocab: usize,
    },
    MxFp4(MxFp4EmbeddingLookup),
    Ggml(GgmlEmbeddingLookup),
    PackedInteger {
        operation: CompressedInt8Embedding,
        vocab: usize,
    },
}

#[derive(Debug)]
pub(super) struct SharedRoutedOutputHead {
    operation: CheckpointProjection,
    tokens: usize,
    hidden: usize,
    vocab: usize,
}

impl SharedRoutedEmbedding {
    pub(super) fn new(
        backend: &CudaBackend,
        hidden: usize,
        vocab: usize,
        weight: &CheckpointProjectionWeight,
    ) -> Result<Self> {
        weight.affine_format(1, hidden, vocab)?;
        match weight {
            CheckpointProjectionWeight::Affine(weight) => {
                let config = weight.infer_config(1, hidden, vocab)?;
                Ok(Self::Affine {
                    operation: backend.prepare_affine_embedding(config, 1.0)?,
                    weight: weight.clone(),
                    vocab,
                })
            },
            CheckpointProjectionWeight::Dense(weight) => Ok(Self::Dense {
                operation: backend.prepare_bf16_embedding(vocab, hidden, 1.0)?,
                weight: weight.clone(),
                vocab,
            }),
            CheckpointProjectionWeight::DirectFp8(_) => Err(Error::InvalidExecutionPlan(
                "shared routed embedding does not support direct FP8",
            )),
            CheckpointProjectionWeight::MxFp4(weight) => {
                weight.prepare_embedding(backend, 1.0).map(Self::MxFp4)
            },
            CheckpointProjectionWeight::MxFp8(_) => {
                Err(Error::InvalidExecutionPlan("shared routed embedding does not support MXFP8"))
            },
            CheckpointProjectionWeight::Ggml(weight) => {
                weight.prepare_embedding(backend, 1.0).map(Self::Ggml)
            },
            CheckpointProjectionWeight::NvFp4(_)
            | CheckpointProjectionWeight::NvFp4WeightOnly(_) => {
                Err(Error::InvalidExecutionPlan("shared routed embedding does not support NVFP4"))
            },
            CheckpointProjectionWeight::PackedInteger(weight) => Ok(Self::PackedInteger {
                operation: packed_integer_embedding(backend, vocab, hidden, 1.0, weight.clone())?,
                vocab,
            }),
        }
    }

    pub(super) fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Affine { operation, weight, .. } => {
                operation.execute_batch(selected, 0, tokens, weight.tensors(), output)
            },
            Self::Dense { operation, weight, .. } => {
                operation.execute_batch(selected, 0, tokens, weight, output)
            },
            Self::PackedInteger { operation, .. } => {
                operation.execute_batch(selected, 0, tokens, output)
            },
            Self::MxFp4(operation) => operation.execute_batch(selected, 0, tokens, output),
            Self::Ggml(operation) => operation.execute_batch(selected, 0, tokens, output),
        }
    }

    pub(super) fn validate_token(&self, token: u32) -> Result<()> {
        let vocab = match self {
            Self::MxFp4(operation) => return operation.validate_token(token),
            Self::Ggml(operation) => return operation.validate_token(token),
            Self::Affine { vocab, .. }
            | Self::Dense { vocab, .. }
            | Self::PackedInteger { vocab, .. } => *vocab,
        };
        if usize::try_from(token)? < vocab {
            Ok(())
        } else {
            Err(Error::InvalidToken { token, vocab })
        }
    }
}

impl SharedRoutedOutputHead {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        hidden: usize,
        vocab: usize,
        weight: &CheckpointProjectionWeight,
    ) -> Result<Self> {
        Ok(Self {
            operation: CheckpointProjection::new(
                backend,
                tokens,
                hidden,
                vocab,
                DenseRole::OutputHead,
                weight,
            )?,
            tokens,
            hidden,
            vocab,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if input.len() != self.tokens * self.hidden || output.len() != self.tokens * self.vocab {
            return Err(Error::InvalidDecoderKernel("shared-routed output-head buffer mismatch"));
        }
        self.operation.execute(input, output)
    }
}

impl super::CudaSharedRoutedModelTemplate {
    pub(super) fn prepare_embedding(&self) -> Result<SharedRoutedEmbedding> {
        SharedRoutedEmbedding::new(
            &self.backend,
            self.decoder.hidden_size,
            self.decoder.vocab_size,
            &self.embedding,
        )
    }

    pub(super) fn prepare_output_head(&self, tokens: usize) -> Result<SharedRoutedOutputHead> {
        SharedRoutedOutputHead::new(
            &self.backend,
            tokens,
            self.decoder.hidden_size,
            self.decoder.vocab_size,
            &self.output,
        )
    }
}
