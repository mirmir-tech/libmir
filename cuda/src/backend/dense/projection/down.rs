use mircuda::{DeviceBuffer, Stream, bf16};

use super::super::{CudaBackend, DenseDownWeight, DenseSwiGluConfig};
use crate::{
    Bf16Projection, DenseExecution, DensePlanRequest, DenseRole, DirectFp8Bf16Linear, Error,
    ExecutionPhase, MxFp4Bf16Linear, MxFp8Bf16Linear, NvFp4Bf16Linear, PackedIntegerBf16Linear,
    ProjectionFormat, Result, backend::linear::AffineProjection,
};

pub(in crate::backend::dense) enum DownProjection {
    Affine(AffineProjection),
    Bf16(Bf16Projection),
    DirectFp8(DirectFp8Bf16Linear),
    MxFp4(MxFp4Bf16Linear),
    MxFp8(MxFp8Bf16Linear),
    PackedInteger(PackedIntegerBf16Linear),
    NvFp4(NvFp4Bf16Linear),
    BlockFp8,
    Fp8Int4,
}

impl DownProjection {
    pub(in crate::backend::dense) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        tokens: usize,
        weight: Option<DenseDownWeight<'_>>,
    ) -> Result<Self> {
        match config.attention.projection_format {
            ProjectionFormat::Affine => {
                let DenseDownWeight::Affine(weight) = weight.ok_or(Error::InvalidExecutionPlan(
                    "affine MLP requires prepared down weight",
                ))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine MLP received non-affine down weight",
                    ));
                };
                let affine = weight.infer_config(
                    1,
                    config.intermediate_size,
                    config.attention.hidden_size,
                )?;
                Ok(Self::Affine(AffineProjection::new(
                    backend,
                    tokens,
                    config.intermediate_size,
                    config.attention.hidden_size,
                    affine.group_size,
                    affine.bits,
                    weight,
                )?))
            },
            ProjectionFormat::Bf16 => Self::new_bf16(backend, config, tokens),
            ProjectionFormat::DirectFp8 => {
                let DenseDownWeight::DirectFp8(weight) = weight
                    .ok_or(Error::InvalidExecutionPlan("direct FP8 MLP requires down weight"))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 MLP received another down weight",
                    ));
                };
                Ok(Self::DirectFp8(weight.prepare(backend, tokens)?))
            },
            ProjectionFormat::MxFp8 => {
                let DenseDownWeight::MxFp8(weight) =
                    weight.ok_or(Error::InvalidExecutionPlan("MXFP8 MLP requires down weight"))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP8 MLP received another down weight",
                    ));
                };
                Ok(Self::MxFp8(weight.prepare(backend, tokens)?))
            },
            ProjectionFormat::MxFp4 => {
                let DenseDownWeight::MxFp4(weight) =
                    weight.ok_or(Error::InvalidExecutionPlan("MXFP4 MLP requires down weight"))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP4 MLP received another down weight",
                    ));
                };
                Ok(Self::MxFp4(weight.prepare(backend, tokens)?))
            },
            ProjectionFormat::PackedInteger => {
                let DenseDownWeight::PackedInteger(weight) = weight.ok_or(
                    Error::InvalidExecutionPlan("packed integer MLP requires prepared down weight"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer MLP received other down weight",
                    ));
                };
                Ok(Self::PackedInteger(PackedIntegerBf16Linear::new(
                    backend,
                    tokens,
                    config.intermediate_size,
                    config.attention.hidden_size,
                    weight,
                )?))
            },
            ProjectionFormat::NvFp4 => {
                let DenseDownWeight::NvFp4(weight) = weight.ok_or(Error::InvalidExecutionPlan(
                    "NVFP4 MLP requires prepared down weight",
                ))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 MLP received non-NVFP4 down weight",
                    ));
                };
                Ok(Self::NvFp4(NvFp4Bf16Linear::from_weight(backend, tokens, weight.clone())?))
            },
        }
    }

    fn new_bf16(backend: &CudaBackend, config: DenseSwiGluConfig, tokens: usize) -> Result<Self> {
        let request = DensePlanRequest {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            role: DenseRole::DenseDown,
            tokens,
            input_features: config.intermediate_size,
            output_features: config.attention.hidden_size,
        };
        Ok(
            match backend
                .execution_planner()
                .plan_dense_with_prepared_weights(request)?
                .execution()
            {
                DenseExecution::BlockFp8Vector => Self::BlockFp8,
                DenseExecution::Fp8Int4Vector => Self::Fp8Int4,
                DenseExecution::Matrix | DenseExecution::Vector | DenseExecution::CublasLt => {
                    Self::Bf16(backend.prepare_bf16_projection(request)?)
                },
            },
        )
    }

    pub(in crate::backend::dense) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: DenseDownWeight<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Affine(operation) => match weight {
                DenseDownWeight::Affine(weight) => operation.execute(input, weight, output),
                _ => Err(Error::InvalidExecutionPlan(
                    "affine down operation received non-affine weight",
                )),
            },
            Self::Bf16(operation) => operation.execute(input, weight.require_bf16()?, output),
            Self::DirectFp8(operation) => match weight {
                DenseDownWeight::DirectFp8(weight) => operation.execute(input, weight, output),
                _ => Err(Error::InvalidExecutionPlan(
                    "direct FP8 down operation received another weight",
                )),
            },
            Self::MxFp8(operation) => match weight {
                DenseDownWeight::MxFp8(weight) => operation.execute(input, weight, output),
                _ => {
                    Err(Error::InvalidExecutionPlan("MXFP8 down operation received another weight"))
                },
            },
            Self::MxFp4(operation) => match weight {
                DenseDownWeight::MxFp4(weight) => operation.execute(input, weight, output),
                _ => {
                    Err(Error::InvalidExecutionPlan("MXFP4 down operation received another weight"))
                },
            },
            Self::PackedInteger(operation) => match weight {
                DenseDownWeight::PackedInteger(weight) => operation.execute(input, weight, output),
                _ => Err(Error::InvalidExecutionPlan(
                    "packed integer down operation received other weight",
                )),
            },
            Self::NvFp4(operation) => match weight {
                DenseDownWeight::NvFp4(_) => operation.execute(input, output),
                _ => Err(Error::InvalidExecutionPlan("NVFP4 down operation received other weight")),
            },
            Self::BlockFp8 => match weight {
                DenseDownWeight::BlockFp8 { quantized, .. } => {
                    quantized.execute(stream, input, output)
                },
                _ => Err(Error::InvalidExecutionPlan("block FP8 down plan lacks quantized weight")),
            },
            Self::Fp8Int4 => match weight {
                DenseDownWeight::Fp8Int4 { quantized, .. } => {
                    quantized.execute(stream, input, output)
                },
                _ => Err(Error::InvalidExecutionPlan(
                    "FP8 plus INT4 down plan lacks quantized weight",
                )),
            },
        }
    }

    pub(in crate::backend::dense) fn execute_gated(
        &mut self,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        activation: crate::GatedActivation,
        weight: DenseDownWeight<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<bool> {
        match (self, weight) {
            (Self::Affine(_), DenseDownWeight::Affine(_))
            | (Self::Bf16(_), DenseDownWeight::Bf16(_))
            | (Self::DirectFp8(_), DenseDownWeight::DirectFp8(_))
            | (Self::MxFp4(_), DenseDownWeight::MxFp4(_))
            | (Self::MxFp8(_), DenseDownWeight::MxFp8(_))
            | (Self::PackedInteger(_), DenseDownWeight::PackedInteger(_))
            | (Self::BlockFp8, DenseDownWeight::BlockFp8 { .. })
            | (Self::Fp8Int4, DenseDownWeight::Fp8Int4 { .. }) => Ok(false),
            (Self::NvFp4(operation), DenseDownWeight::NvFp4(_)) => {
                operation.execute_gated(gate, up, activation.into(), output)?;
                Ok(true)
            },
            _ => Err(Error::InvalidExecutionPlan(
                "down projection operation and weight format differ",
            )),
        }
    }
}
