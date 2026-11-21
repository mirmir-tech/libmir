//! GGML block weights read in place from GGUF files. Each tensor keeps its
//! own GGML type, so one layer may mix `Q4_K` and `Q6_K` projections.

mod linear;
mod scratch;

pub use linear::GgmlBf16Linear;
use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{TensorBinding, TensorStorage};

use super::CudaBackend;
use crate::{
    CudaTensorDType, CudaTensorSet, DenseRole, Error, Result,
    kernels::{GgmlEmbedding, GgmlEmbeddingOperands, GgmlKernelType},
};

/// One validated GGML projection matrix, `[output, input]` in logical shape.
#[derive(Clone, Debug)]
pub struct GgmlCheckpointWeight {
    blocks: DeviceBuffer<u8>,
    kind: GgmlKernelType,
    input_features: usize,
    output_features: usize,
}

/// Prepared selected-row GGML embedding lookup.
#[derive(Debug)]
pub struct GgmlEmbeddingLookup {
    operation: GgmlEmbedding,
    stream: Stream,
    weight: GgmlCheckpointWeight,
}

impl GgmlCheckpointWeight {
    pub fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        let unsupported =
            |reason: &str| Error::UnsupportedDecoderLayer(format!("{}: {reason}", binding.source));
        let TensorStorage::Ggml { format, .. } = &binding.storage else {
            return Err(unsupported("requires GGML block storage"));
        };
        let kind = GgmlKernelType::new(format.ggml_type)?;
        // A matrix bank `[matrices, out, in]` (stacked experts) binds as one
        // matrix of `matrices * out` rows; its consumers address the banks.
        let (input_features, output_features) = match binding.logical_shape.as_deref() {
            Some(&[output, input]) => (input, output),
            Some(&[matrices, output, input]) => (input, matrices * output),
            _ => return Err(unsupported("GGML projection is not a matrix or matrix bank")),
        };
        let (input_features, output_features) = (&input_features, &output_features);
        let tensor = tensors
            .get(&binding.source)
            .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
        let blocks =
            tensor
                .as_u8()
                .filter(|_| tensor.dtype() == CudaTensorDType::U8)
                .ok_or_else(|| Error::DTypeMismatch {
                    name: binding.source.clone(),
                    expected: "GGML blocks",
                })?;
        let expected = kind.bytes(*output_features, *input_features)?;
        if blocks.len() != expected {
            return Err(Error::InvalidTensorSize {
                name: binding.source.clone(),
                expected,
                actual: blocks.len(),
            });
        }
        Ok(Self {
            blocks: blocks.clone(),
            kind,
            input_features: *input_features,
            output_features: *output_features,
        })
    }

    #[must_use]
    pub const fn kind(&self) -> GgmlKernelType {
        self.kind
    }

    /// Output rows, across every matrix of a bank.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.output_features
    }

    #[must_use]
    pub const fn blocks(&self) -> &DeviceBuffer<u8> {
        &self.blocks
    }

    pub fn prepare(
        &self,
        backend: &CudaBackend,
        tokens: usize,
        role: DenseRole,
    ) -> Result<GgmlBf16Linear> {
        GgmlBf16Linear::new(backend, tokens, role, self.input_features, self.output_features)
    }

    /// The whole matrix in BF16, `[output, input]` row-major, for consumers
    /// that read weights directly (latent attention's absorption GEMMs).
    pub(crate) fn dequantize_bf16(&self, backend: &CudaBackend) -> Result<DeviceBuffer<bf16>> {
        let operation = crate::kernels::GgmlDequantize::compile(
            &backend.inner.compiler,
            self.kind,
            self.output_features,
            self.input_features,
        )?;
        let mut output =
            backend.inner.pool.allocate(&backend.inner.stream, operation.elements()?)?;
        operation.execute(&backend.inner.stream, &self.blocks, &mut output)?;
        Ok(output)
    }

    pub(crate) fn validate(&self, input: usize, output: usize) -> Result<()> {
        if self.input_features == input && self.output_features == output {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("GGML checkpoint geometry differs"))
        }
    }

    pub fn prepare_embedding(
        &self,
        backend: &CudaBackend,
        output_scale: f32,
    ) -> Result<GgmlEmbeddingLookup> {
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
