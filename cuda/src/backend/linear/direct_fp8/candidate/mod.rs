use std::sync::Arc;

use mircuda::{DeviceBuffer, ScaledFp8Scale, ScaledFp8Tile, Stream, bf16};

mod admission;

pub(super) use admission::tensor_core_admitted;
use admission::{bias, cublaslt_admitted};

use super::{
    CudaBackend, CudaTensor, DirectE5M2WeightOnlyTensorCoreLinear, DirectFp8Activation,
    DirectFp8CachedLinear, DirectFp8CheckpointWeight, DirectFp8CublasLtLinear, DirectFp8Format,
    DirectFp8Linear, DirectFp8Scale, DirectFp8Scales, DirectFp8Spec, DirectFp8TensorCoreLinear,
    Error, Result, unsupported,
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
    PortableCached(DirectFp8CachedLinear),
    TensorCore(Arc<DirectFp8TensorCoreLinear>),
    CublasLt(Arc<DirectFp8CublasLtLinear>),
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
        if matches!(
            execution,
            DirectFp8ProjectionExecution::TensorCore | DirectFp8ProjectionExecution::TensorCoreWide
        ) && !tensor_core_admitted(backend, spec, tensor_core_scale)
        {
            return Err(Error::InvalidExecutionPlan("direct FP8 Tensor Core is unavailable"));
        }
        let operation = match execution {
            DirectFp8ProjectionExecution::Portable => {
                Operation::Portable(DirectFp8Linear::compile(&backend.inner.compiler, spec)?)
            },
            DirectFp8ProjectionExecution::PortableCached => {
                if tensor_core_scale != Some(ScaledFp8Scale::F32) {
                    return Err(Error::InvalidExecutionPlan(
                        "cached direct FP8 requires F32 scales",
                    ));
                }
                Operation::PortableCached(DirectFp8CachedLinear::compile(
                    &backend.inner.compiler,
                    spec,
                )?)
            },
            DirectFp8ProjectionExecution::TensorCore
            | DirectFp8ProjectionExecution::TensorCoreWide => {
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
                        match execution {
                            DirectFp8ProjectionExecution::TensorCore => ScaledFp8Tile::M16N64K128,
                            DirectFp8ProjectionExecution::TensorCoreWide => {
                                ScaledFp8Tile::M16N128K64
                            },
                            _ => unreachable!(),
                        },
                    )?))
                }
            },
            DirectFp8ProjectionExecution::CublasLt => {
                if !cublaslt_admitted(spec, tensor_core_scale, has_bias) {
                    return Err(Error::InvalidExecutionPlan("direct FP8 cuBLASLt is unavailable"));
                }
                Operation::CublasLt(Arc::new(DirectFp8CublasLtLinear::prepare(
                    &backend.inner.compiler,
                    &backend.inner.context,
                    &backend.inner.pool,
                    &backend.inner.stream,
                    spec,
                )?))
            },
        };
        Ok(Self { execution, operation })
    }

    #[allow(clippy::too_many_lines)]
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
                Operation::PortableCached(operation) => Ok(operation.execute(
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
                Operation::CublasLt(operation) => Ok(operation.execute(
                    stream,
                    input,
                    weight_buffer,
                    scales,
                    weight.input_scale.as_ref().and_then(CudaTensor::as_f32).ok_or(
                        Error::InvalidExecutionPlan("direct FP8 cuBLASLt input scale is missing"),
                    )?,
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
                Operation::PortableCached(_) => {
                    Err(Error::InvalidExecutionPlan("cached direct FP8 requires F32 scales"))
                },
                Operation::TensorCore(operation) => Ok(operation.execute_bf16_scales(
                    stream,
                    input,
                    weight_buffer,
                    scales,
                    weight.input_scale.as_ref().and_then(CudaTensor::as_bf16),
                    bias,
                    output,
                )?),
                Operation::CublasLt(_) => {
                    Err(Error::InvalidExecutionPlan("direct FP8 cuBLASLt requires F32 scales"))
                },
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
            Operation::PortableCached(_) => {
                Err(unsupported(weight.weight.name(), "cached direct FP8 requires scaled E4M3"))
            },
            Operation::TensorCore(_) => {
                Err(unsupported(weight.weight.name(), "cannot use an identity Tensor Core scale"))
            },
            Operation::CublasLt(_) => {
                Err(unsupported(weight.weight.name(), "cannot use an identity cuBLASLt FP8 scale"))
            },
            Operation::E5M2TensorCore(operation) => {
                Ok(operation.execute(stream, input, weight_buffer, bias, output)?)
            },
        }
    }
}
