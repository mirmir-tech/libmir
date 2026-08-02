use mircuda::{DeviceBuffer, ScaledFp8Scale, bf16};

use super::CudaBackend;
use crate::{
    CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    kernels::{
        DirectE5M2WeightOnlyTensorCoreLinear, DirectFp8Activation, DirectFp8Embedding,
        DirectFp8EmbeddingBatch, DirectFp8EmbeddingSpec, DirectFp8Format, DirectFp8Linear,
        DirectFp8Scale, DirectFp8Scales, DirectFp8Spec, DirectFp8TensorCoreLinear,
    },
};

mod candidate;
mod contract;
mod embedding;
use contract::{execution_contract, unsupported};
pub use embedding::DirectFp8EmbeddingLookup;
mod load;
mod storage;
mod tuning;
use storage::{identity_scale, tensor};
#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
/// Validated model-owned direct E4M3 or E5M2 checkpoint tensors.
pub struct DirectFp8CheckpointWeight {
    weight: CudaTensor,
    scales: Option<CudaTensor>,
    input_scale: Option<CudaTensor>,
    bias: Option<CudaTensor>,
    input_features: usize,
    output_features: usize,
    format: DirectFp8Format,
    scale: DirectFp8Scale,
    inverse_scale: bool,
    activation: DirectFp8Activation,
}

#[derive(Clone, Debug)]
/// Prepared BF16-input projection retaining direct checkpoint tensors.
pub struct DirectFp8Bf16Linear {
    operation: candidate::Candidate,
    stream: mircuda::Stream,
    spec: DirectFp8Spec,
    scale_dtype: Option<CudaTensorDType>,
    identity_scale: Option<DeviceBuffer<f32>>,
    has_bias: bool,
}

impl DirectFp8CheckpointWeight {
    pub fn prepare(&self, backend: &CudaBackend, tokens: usize) -> Result<DirectFp8Bf16Linear> {
        let spec = DirectFp8Spec::new_with_format(
            self.format,
            tokens,
            self.input_features,
            self.output_features,
            self.scale,
            self.inverse_scale,
            self.activation,
        )?;
        let tensor_core_scale = match self.scales.as_ref().map(CudaTensor::dtype) {
            Some(CudaTensorDType::F32) => Some(ScaledFp8Scale::F32),
            Some(CudaTensorDType::Bf16) => Some(ScaledFp8Scale::Bf16),
            _ => None,
        };
        let identity_scale = (self.scales.is_none()
            || (self.activation != DirectFp8Activation::StaticE4M3Tensor
                && self
                    .scales
                    .as_ref()
                    .is_some_and(|value| value.dtype() == CudaTensorDType::F32)))
        .then(|| identity_scale(backend))
        .transpose()?;
        let operation =
            tuning::prepare(backend, self, spec, tensor_core_scale, identity_scale.as_ref())?;
        Ok(DirectFp8Bf16Linear {
            operation,
            stream: backend.inner.stream.clone(),
            spec,
            scale_dtype: self.scales.as_ref().map(CudaTensor::dtype),
            identity_scale,
            has_bias: self.bias.is_some(),
        })
    }

    pub(crate) fn validate(&self, input: usize, output: usize) -> Result<()> {
        if self.input_features == input && self.output_features == output {
            Ok(())
        } else {
            Err(Error::InvalidLinearWeight {
                name: self.weight.name().into(),
                expected: [output, input],
                actual: self.weight.shape().to_vec(),
            })
        }
    }
}

impl DirectFp8CheckpointWeight {
    const fn format_name(&self) -> &'static str {
        match self.format {
            DirectFp8Format::E4M3 => "F8_E4M3",
            DirectFp8Format::E5M2 => "F8_E5M2",
        }
    }
}

impl DirectFp8Bf16Linear {
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &DirectFp8CheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate_weight(weight)?;
        self.operation
            .execute(&self.stream, input, weight, self.identity_scale.as_ref(), output)
    }

    fn validate_weight(&self, weight: &DirectFp8CheckpointWeight) -> Result<()> {
        let compatible = self.spec.input_features == weight.input_features
            && self.spec.output_features == weight.output_features
            && self.spec.format == weight.format
            && self.spec.scale == weight.scale
            && self.spec.inverse_scale == weight.inverse_scale
            && self.spec.activation == weight.activation
            && self.scale_dtype == weight.scales.as_ref().map(CudaTensor::dtype)
            && (self.spec.activation == DirectFp8Activation::StaticE4M3Tensor)
                == weight.input_scale.is_some()
            && self.has_bias == weight.bias.is_some();
        if compatible {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan(
                "direct FP8 plan and late-bound weight contract differ",
            ))
        }
    }
}
