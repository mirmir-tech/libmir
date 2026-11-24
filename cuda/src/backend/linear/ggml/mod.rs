//! GGML block weights read in place from GGUF files. Each tensor keeps its
//! own GGML type, so one layer may mix `Q4_K` and `Q6_K` projections.

mod embedding;
mod linear;
mod scale;
mod scratch;
#[cfg(test)]
pub(in crate::backend) mod tests;

pub use embedding::GgmlEmbeddingLookup;
pub use linear::GgmlBf16Linear;
use mircuda::{DeviceBuffer, bf16};
use models::weights::{TensorBinding, TensorStorage};

use super::CudaBackend;
use crate::{CudaTensorDType, CudaTensorSet, DenseRole, Error, Result, kernels::GgmlKernelType};

/// One validated GGML projection matrix, `[output, input]` in logical shape.
#[derive(Clone, Debug)]
pub struct GgmlCheckpointWeight {
    blocks: DeviceBuffer<u8>,
    kind: GgmlKernelType,
    input_features: usize,
    output_features: usize,
    /// Matrices of a bank (experts); one for a single matrix.
    matrices: usize,
    /// F32 output scales, one per matrix (llama.cpp's `.scale`).
    scale: Option<DeviceBuffer<f32>>,
    /// An output bias the projection adds itself (Qwen2, GLM-4 attention);
    /// consumers that add their binding's bias separately leave it unset.
    bias: Option<DeviceBuffer<bf16>>,
}

impl GgmlCheckpointWeight {
    pub fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        let unsupported =
            |reason: &str| Error::UnsupportedDecoderLayer(format!("{}: {reason}", binding.source));
        let TensorStorage::Ggml { format, companion } = &binding.storage else {
            return Err(unsupported("requires GGML block storage"));
        };
        let kind = GgmlKernelType::new(format.ggml_type)?;
        // A matrix bank `[matrices, out, in]` (stacked experts) binds as one
        // matrix of `matrices * out` rows; its consumers address the banks.
        let (input_features, output_features, matrices) = match binding.logical_shape.as_deref() {
            Some(&[output, input]) => (input, output, 1),
            Some(&[matrices, output, input]) => (input, matrices * output, matrices),
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
        let scale = companion
            .scale()
            .map(|name| output_scale(tensors, name, matrices))
            .transpose()?;
        Ok(Self {
            blocks: blocks.clone(),
            kind,
            input_features: *input_features,
            output_features: *output_features,
            matrices,
            scale,
            bias: None,
        })
    }

    /// [`Self::load_binding`] that adds the binding's dense output bias after
    /// its matmul.
    pub fn load_binding_with_bias(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
    ) -> Result<Self> {
        let mut weight = Self::load_binding(tensors, binding)?;
        let TensorStorage::Ggml {
            companion: models::weights::GgmlCompanion::Bias(name),
            ..
        } = &binding.storage
        else {
            return Ok(weight);
        };
        let tensor = tensors.get(name).ok_or_else(|| Error::MissingTensor(name.clone()))?;
        let values = tensor.as_bf16().filter(|values| values.len() == weight.output_features);
        let values = values.ok_or_else(|| Error::DTypeMismatch {
            name: name.clone(),
            expected: "a BF16 bias of the projection's output width",
        })?;
        weight.bias = Some(values.clone());
        Ok(weight)
    }

    /// A weight of `matrices` stacked `[rows, input]` matrices of `kind`
    /// blocks, for tests that build their own blocks.
    #[cfg(test)]
    pub(crate) fn from_blocks(
        blocks: DeviceBuffer<u8>,
        kind: GgmlKernelType,
        (matrices, rows, input_features): (usize, usize, usize),
        scale: Option<DeviceBuffer<f32>>,
    ) -> Self {
        Self {
            blocks,
            kind,
            input_features,
            output_features: matrices * rows,
            matrices,
            scale,
            bias: None,
        }
    }

    /// The F32 output scales, one per matrix, when the projection has them.
    #[must_use]
    pub const fn output_scale(&self) -> Option<&DeviceBuffer<f32>> {
        self.scale.as_ref()
    }

    /// Input features of every row.
    #[must_use]
    pub const fn input_features(&self) -> usize {
        self.input_features
    }

    /// The same projection over `blocks` of the same type and geometry,
    /// e.g. its rows reordered.
    pub(crate) fn with_blocks(&self, blocks: DeviceBuffer<u8>) -> Result<Self> {
        if blocks.len() != self.blocks.len() {
            return Err(Error::InvalidExecutionPlan("replacement GGML blocks differ in size"));
        }
        Ok(Self { blocks, ..self.clone() })
    }

    /// Matrices of the bank; one for a single matrix.
    #[must_use]
    pub const fn matrices(&self) -> usize {
        self.matrices
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
        let stream = &backend.inner.stream;
        let mut output = backend.inner.pool.allocate(stream, operation.elements()?)?;
        operation.execute(stream, &self.blocks, &mut output)?;
        match (&self.scale, self.matrices) {
            (None, _) => {},
            (Some(scale), 1) => {
                let scaling = crate::kernels::GgmlOutputScaling::compile(&backend.inner.compiler)?;
                let elements = output.len();
                scaling.matrix(stream, &mut output, scale, elements)?;
            },
            (Some(_), _) => {
                return Err(Error::InvalidExecutionPlan("dequantized GGML banks take no scales"));
            },
        }
        Ok(output)
    }

    pub(crate) fn validate(&self, input: usize, output: usize) -> Result<()> {
        if self.input_features == input && self.output_features == output {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("GGML checkpoint geometry differs"))
        }
    }
}

/// The `matrices` F32 values of output scale `name`; kept F32 at upload.
fn output_scale(tensors: &CudaTensorSet, name: &str, matrices: usize) -> Result<DeviceBuffer<f32>> {
    let tensor = tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
    let values =
        tensor
            .as_f32()
            .filter(|_| tensor.dtype() == CudaTensorDType::F32)
            .ok_or_else(|| Error::DTypeMismatch {
                name: name.into(),
                expected: "F32 output scale",
            })?;
    if values.len() != matrices {
        return Err(Error::InvalidTensorSize {
            name: name.into(),
            expected: matrices,
            actual: values.len(),
        });
    }
    Ok(values.clone())
}
