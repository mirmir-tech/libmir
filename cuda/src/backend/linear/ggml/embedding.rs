//! Selected-row lookups of a GGML embedding.

use mircuda::{DeviceBuffer, Stream, bf16};

use super::GgmlCheckpointWeight;
use crate::{
    CudaBackend, Error, Result,
    kernels::{GgmlEmbedding, GgmlEmbeddingOperands},
};

/// Prepared selected-row GGML embedding lookup.
#[derive(Debug)]
pub struct GgmlEmbeddingLookup {
    operation: GgmlEmbedding,
    stream: Stream,
    weight: GgmlCheckpointWeight,
}

impl GgmlCheckpointWeight {
    pub fn prepare_embedding(
        &self,
        backend: &CudaBackend,
        output_scale: f32,
    ) -> Result<GgmlEmbeddingLookup> {
        // llama.cpp scales projection products; an embedding row has none.
        if self.scale.is_some() {
            return Err(Error::InvalidExecutionPlan("GGML embeddings take no output scale"));
        }
        Ok(GgmlEmbeddingLookup {
            operation: GgmlEmbedding::compile(
                &backend.inner.compiler,
                self.kind,
                self.output_features,
                self.input_features,
                output_scale,
            )?,
            stream: backend.inner.stream.clone(),
            weight: self.clone(),
        })
    }
}

impl GgmlEmbeddingLookup {
    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.operation.execute(
            &self.stream,
            &mut GgmlEmbeddingOperands {
                weight: &self.weight.blocks,
                selected,
                output,
            },
            selected_start,
            tokens,
        )
    }

    pub fn validate_token(&self, token: u32) -> Result<()> {
        if usize::try_from(token)? < self.weight.output_features {
            Ok(())
        } else {
            Err(Error::InvalidToken {
                token,
                vocab: self.weight.output_features,
            })
        }
    }
}
