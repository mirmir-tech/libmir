use mircuda::{DeviceBuffer, bf16};

use crate::{
    AffineQuantizedEmbedding, Bf16Embedding, CudaBackend, DirectFp8EmbeddingLookup, Error,
    MxFp4EmbeddingLookup, MxFp8EmbeddingLookup, Result,
    backend::linear::{
        CheckpointProjectionWeight, CompressedInt8Embedding, packed_integer_embedding,
    },
};

#[derive(Clone)]
pub(in crate::backend::model) struct ModelEmbeddingTemplate {
    weight: CheckpointProjectionWeight,
    vocab: usize,
    hidden: usize,
    scale: f32,
}

pub(in crate::backend::model) enum ModelEmbedding {
    Affine {
        operation: AffineQuantizedEmbedding,
        weight: crate::AffineQuantizedWeight,
    },
    Dense {
        operation: Bf16Embedding,
        weight: crate::CudaTensor,
    },
    MxFp4(MxFp4EmbeddingLookup),
    MxFp8(MxFp8EmbeddingLookup),
    DirectFp8(DirectFp8EmbeddingLookup),
    PackedInteger(CompressedInt8Embedding),
}

impl ModelEmbeddingTemplate {
    pub(in crate::backend::model) fn new(
        weight: CheckpointProjectionWeight,
        vocab: usize,
        hidden: usize,
        scale: f32,
    ) -> Result<Self> {
        weight.affine_format(1, hidden, vocab)?;
        Ok(Self { weight, vocab, hidden, scale })
    }

    pub(in crate::backend::model) fn instantiate(
        &self,
        backend: &CudaBackend,
    ) -> Result<ModelEmbedding> {
        match &self.weight {
            CheckpointProjectionWeight::Affine(weight) => {
                let config = weight.infer_config(1, self.hidden, self.vocab)?;
                Ok(ModelEmbedding::Affine {
                    operation: backend.prepare_affine_embedding(config, self.scale)?,
                    weight: weight.clone(),
                })
            },
            CheckpointProjectionWeight::Dense(weight) => Ok(ModelEmbedding::Dense {
                operation: backend.prepare_bf16_embedding(self.vocab, self.hidden, self.scale)?,
                weight: weight.clone(),
            }),
            CheckpointProjectionWeight::DirectFp8(weight) => {
                weight.prepare_embedding(backend, self.scale).map(ModelEmbedding::DirectFp8)
            },
            CheckpointProjectionWeight::MxFp4(weight) => {
                weight.prepare_embedding(backend, self.scale).map(ModelEmbedding::MxFp4)
            },
            CheckpointProjectionWeight::MxFp8(weight) => {
                weight.prepare_embedding(backend, self.scale).map(ModelEmbedding::MxFp8)
            },
            CheckpointProjectionWeight::NvFp4(_)
            | CheckpointProjectionWeight::NvFp4WeightOnly(_) => {
                Err(Error::InvalidExecutionPlan("CUDA embedding does not support NVFP4"))
            },
            CheckpointProjectionWeight::PackedInteger(weight) => packed_integer_embedding(
                backend,
                self.vocab,
                self.hidden,
                self.scale,
                weight.clone(),
            )
            .map(ModelEmbedding::PackedInteger),
        }
    }
}

impl ModelEmbedding {
    pub(in crate::backend::model) fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Affine { operation, weight } => {
                operation.execute_batch(selected, selected_start, tokens, weight.tensors(), output)
            },
            Self::Dense { operation, weight } => {
                operation.execute_batch(selected, selected_start, tokens, weight, output)
            },
            Self::PackedInteger(operation) => {
                operation.execute_batch(selected, selected_start, tokens, output)
            },
            Self::MxFp8(operation) => {
                operation.execute_batch(selected, selected_start, tokens, output)
            },
            Self::MxFp4(operation) => {
                operation.execute_batch(selected, selected_start, tokens, output)
            },
            Self::DirectFp8(operation) => {
                operation.execute_batch(selected, selected_start, tokens, output)
            },
        }
    }

    pub(in crate::backend::model) fn execute(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_index: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_batch(selected, selected_index, 1, output)
    }

    pub(in crate::backend::model) fn validate_token(&self, token: u32) -> Result<()> {
        match self {
            Self::Affine { operation, .. } => operation.validate_token(token),
            Self::Dense { operation, .. } => operation.validate_token(token),
            Self::PackedInteger(operation) => operation.validate_token(token),
            Self::MxFp8(operation) => operation.validate_token(token),
            Self::MxFp4(operation) => operation.validate_token(token),
            Self::DirectFp8(operation) => operation.validate_token(token),
        }
    }
}
