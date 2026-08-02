use std::sync::Arc;

use mircuda::{DeviceBuffer, ScaledFp8Scale, Stream, bf16};

use super::{
    CudaBackend, CudaTensor, DirectE5M2WeightOnlyTensorCoreLinear, DirectFp8Activation,
    DirectFp8CheckpointWeight, DirectFp8Format, DirectFp8Linear, DirectFp8Scale, DirectFp8Scales,
    DirectFp8Spec, DirectFp8TensorCoreLinear, Error, Result, unsupported,
};
use crate::backend::tuning::DirectFp8ProjectionExecution;

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub(super) execution: DirectFp8ProjectionExecution,
    operation: Operation,
}

#[derive(Clone, Debug)]
enum Operation {
    Portable(DirectFp8Linear),
    TensorCore(Arc<DirectFp8TensorCoreLinear>),
    E5M2TensorCore(DirectE5M2WeightOnlyTensorCoreLinear),
}

impl Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        spec: DirectFp8Spec,
        tensor_core_scale: Option<ScaledFp8Scale>,
        has_bias: bool,
        execution: DirectFp8ProjectionExecution,
    ) -> Result<Self> {
        if execution == DirectFp8ProjectionExecution::TensorCore
            && !tensor_core_admitted(backend, spec, tensor_core_scale)
        {
            return Err(Error::InvalidExecutionPlan("direct FP8 Tensor Core is unavailable"));
        }
        let operation = match execution {
            DirectFp8ProjectionExecution::Portable => {
                Operation::Portable(DirectFp8Linear::compile(&backend.inner.compiler, spec)?)
            },
            DirectFp8ProjectionExecution::TensorCore => {
                if spec.format == DirectFp8Format::E5M2 {
                    Operation::E5M2TensorCore(DirectE5M2WeightOnlyTensorCoreLinear::compile(
                        &backend.inner.compiler,
                        spec,
                    )?)
                } else {
                    Operation::TensorCore(Arc::new(DirectFp8TensorCoreLinear::prepare(
                        &backend.inner.compiler,
                        &backend.inner.context,
                        &backend.inner.pool,
                        &backend.inner.stream,
                        spec,
                        tensor_core_scale.ok_or(Error::InvalidExecutionPlan(
                            "direct FP8 Tensor Core scale dtype is unavailable",
                        ))?,
                        has_bias,
                    )?))
                }
            },
        };
        Ok(Self { execution, operation })
    }

    pub(super) fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DirectFp8CheckpointWeight,
        identity_scale: Option<&DeviceBuffer<f32>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let (weight_buffer, expected) = match weight.format {
            DirectFp8Format::E4M3 => (weight.weight.as_f8_e4m3(), "F8_E4M3"),
            DirectFp8Format::E5M2 => (weight.weight.as_f8_e5m2(), "F8_E5M2"),
        };
        let weight_buffer = weight_buffer.ok_or_else(|| Error::DTypeMismatch {
            name: weight.weight.name().into(),
            expected,
        })?;
        let bias = bias(weight)?;
        if let Operation::E5M2TensorCore(operation) = &self.operation {
            if weight.scales.is_some() {
                return Err(unsupported(
                    weight.weight.name(),
                    "scaled E5M2 cannot use the weight-only Tensor Core path",
                ));
            }
            return operation.execute(stream, input, weight_buffer, bias, output);
        }
        if let Some(scales) = weight.scales.as_ref().and_then(CudaTensor::as_f32) {
            let input_scale = weight
                .input_scale
                .as_ref()
                .and_then(CudaTensor::as_f32)
                .or(identity_scale)
                .ok_or_else(|| Error::DTypeMismatch {
                    name: weight.weight.name().into(),
                    expected: "F32 activation scale",
                })?;
            return match &self.operation {
                Operation::Portable(operation) => Ok(operation.execute(
                    stream,
                    input,
                    weight_buffer,
                    DirectFp8Scales { weight: scales, activation: input_scale },
                    bias,
                    output,
                )?),
                Operation::TensorCore(operation) => Ok(operation.execute_f32_scales(
                    stream,
                    input,
                    weight_buffer,
                    scales,
                    weight.input_scale.as_ref().and_then(CudaTensor::as_f32),
                    bias,
                    output,
                )?),
                Operation::E5M2TensorCore(_) => Err(Error::InvalidExecutionPlan(
                    "scaled E5M2 reached the weight-only Tensor Core path",
                )),
            };
        }
        if let Some(scales) = weight.scales.as_ref().and_then(CudaTensor::as_bf16) {
            let input_scale =
                weight.input_scale.as_ref().and_then(CudaTensor::as_bf16).unwrap_or(scales);
            return match &self.operation {
                Operation::Portable(operation) => Ok(operation.execute_bf16_scales(
                    stream,
                    input,
                    weight_buffer,
                    DirectFp8Scales { weight: scales, activation: input_scale },
                    bias,
                    output,
                )?),
                Operation::TensorCore(operation) => Ok(operation.execute_bf16_scales(
                    stream,
                    input,
                    weight_buffer,
                    scales,
                    weight.input_scale.as_ref().and_then(CudaTensor::as_bf16),
                    bias,
                    output,
                )?),
                Operation::E5M2TensorCore(_) => Err(Error::InvalidExecutionPlan(
                    "scaled E5M2 reached the weight-only Tensor Core path",
                )),
            };
        }
        let scales = identity_scale.ok_or_else(|| Error::DTypeMismatch {
            name: weight.weight.name().into(),
            expected: "BF16 or F32 scale, or unscaled FP8",
        })?;
        match &self.operation {
            Operation::Portable(operation) => Ok(operation.execute(
                stream,
                input,
                weight_buffer,
                DirectFp8Scales { weight: scales, activation: scales },
                bias,
                output,
            )?),
            Operation::TensorCore(_) => {
                Err(unsupported(weight.weight.name(), "cannot use an identity Tensor Core scale"))
            },
            Operation::E5M2TensorCore(operation) => {
                Ok(operation.execute(stream, input, weight_buffer, bias, output)?)
            },
        }
    }
}

fn bias(weight: &DirectFp8CheckpointWeight) -> Result<Option<&DeviceBuffer<bf16>>> {
    weight
        .bias
        .as_ref()
        .map(|value| {
            value.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                name: value.name().into(),
                expected: "BF16",
            })
        })
        .transpose()
}

pub(super) fn tensor_core_admitted(
    backend: &CudaBackend,
    spec: DirectFp8Spec,
    scale: Option<ScaledFp8Scale>,
) -> bool {
    let scaled_e4m3 = scale.is_some()
        && spec.format == DirectFp8Format::E4M3
        && matches!(
            (spec.scale, spec.activation),
            (DirectFp8Scale::OutputChannel, DirectFp8Activation::DynamicE4M3Token)
                | (
                    DirectFp8Scale::Tensor | DirectFp8Scale::OutputChannel,
                    DirectFp8Activation::StaticE4M3Tensor
                )
        );
    let weight_only_e5m2 = scale.is_none()
        && spec.format == DirectFp8Format::E5M2
        && spec.scale == DirectFp8Scale::Tensor
        && spec.activation == DirectFp8Activation::Bf16
        && spec.input_features.is_multiple_of(16)
        && spec.output_features.is_multiple_of(16);
    backend.inner.device.compute_capability.0 == 12
        && !spec.inverse_scale
        && (scaled_e4m3 || weight_only_e5m2)
}
