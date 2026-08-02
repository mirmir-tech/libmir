use mircuda::{DeviceBuffer, bf16};

use super::{
    CudaBackend, CudaTensor, DirectFp8CheckpointWeight, DirectFp8Embedding,
    DirectFp8EmbeddingBatch, DirectFp8EmbeddingSpec, DirectFp8Format, Error, Result,
    identity_scale,
};

#[derive(Debug)]
/// Prepared selected-row direct FP8 embedding lookup.
pub struct DirectFp8EmbeddingLookup {
    operation: DirectFp8Embedding,
    stream: mircuda::Stream,
    weight: DirectFp8CheckpointWeight,
    identity_scale: Option<DeviceBuffer<f32>>,
}

impl DirectFp8CheckpointWeight {
    /// Prepares selected-row dequantization without expanding the embedding
    /// table.
    pub fn prepare_embedding(
        &self,
        backend: &CudaBackend,
        output_scale: f32,
    ) -> Result<DirectFp8EmbeddingLookup> {
        if self.format != DirectFp8Format::E5M2 {
            return Err(Error::InvalidExecutionPlan(
                "selected-row direct FP8 embedding requires E5M2 storage",
            ));
        }
        if self.bias.is_some() {
            return Err(Error::InvalidExecutionPlan(
                "direct FP8 embedding does not accept projection bias",
            ));
        }
        let operation = DirectFp8Embedding::compile(
            &backend.inner.compiler,
            DirectFp8EmbeddingSpec {
                format: self.format,
                vocab: self.output_features,
                hidden: self.input_features,
                scale: self.scale,
                inverse_scale: self.inverse_scale,
                output_scale,
            },
        )?;
        let identity_scale = self.scales.is_none().then(|| identity_scale(backend)).transpose()?;
        Ok(DirectFp8EmbeddingLookup {
            operation,
            stream: backend.inner.stream.clone(),
            weight: self.clone(),
            identity_scale,
        })
    }
}

impl DirectFp8EmbeddingLookup {
    /// Enqueues selected checkpoint rows as BF16 embeddings.
    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let weight = match self.weight.format {
            DirectFp8Format::E4M3 => self.weight.weight.as_f8_e4m3(),
            DirectFp8Format::E5M2 => self.weight.weight.as_f8_e5m2(),
        }
        .ok_or_else(|| Error::DTypeMismatch {
            name: self.weight.weight.name().into(),
            expected: self.weight.format_name(),
        })?;
        if let Some(scales) = self.weight.scales.as_ref().and_then(CudaTensor::as_f32) {
            return self.operation.execute_f32_scales(
                &self.stream,
                weight,
                scales,
                DirectFp8EmbeddingBatch { selected, selected_start, tokens },
                output,
            );
        }
        if let Some(scales) = self.weight.scales.as_ref().and_then(CudaTensor::as_bf16) {
            return self.operation.execute_bf16_scales(
                &self.stream,
                weight,
                scales,
                DirectFp8EmbeddingBatch { selected, selected_start, tokens },
                output,
            );
        }
        self.operation.execute_f32_scales(
            &self.stream,
            weight,
            self.identity_scale.as_ref().ok_or(Error::InvalidExecutionPlan(
                "unscaled direct FP8 embedding identity is missing",
            ))?,
            DirectFp8EmbeddingBatch { selected, selected_start, tokens },
            output,
        )
    }

    /// Validates one token identifier before device execution.
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
