use mircuda::{DeviceBuffer, bf16};

use super::{CudaBackend, DenseDownWeight, DenseGateUpWeights, DenseSwiGluConfig};
use crate::{
    Bf16Linear, Bf16LinearPair, CompressedInt8Bf16Linear, CudaTensor, Error, ExecutionPhase,
    NvFp4Bf16Linear, NvFp4Bf16Pack, ProjectionFormat, Result, RmsNormBf16,
};

pub(super) enum GateUpProjection {
    Bf16(Bf16LinearPair),
    Int8(Box<[CompressedInt8Bf16Linear; 2]>),
    NvFp4(Box<NvFp4Bf16Pack<2>>),
}

pub(super) enum DownProjection {
    Bf16(Bf16Linear),
    Int8(CompressedInt8Bf16Linear),
    NvFp4(NvFp4Bf16Linear),
}

pub(super) struct GateUpBuffers<'a> {
    pub(super) normalized: &'a mut DeviceBuffer<bf16>,
    pub(super) packed: &'a mut DeviceBuffer<bf16>,
    pub(super) separate: &'a mut [DeviceBuffer<bf16>; 2],
}

impl GateUpProjection {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        tokens: usize,
        weights: Option<DenseGateUpWeights<'_>>,
    ) -> Result<Self> {
        let phase = if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        match config.attention.projection_format {
            ProjectionFormat::Bf16 => Ok(Self::Bf16(Bf16LinearPair::new(
                backend,
                phase,
                tokens,
                config.attention.hidden_size,
                config.intermediate_size,
            )?)),
            ProjectionFormat::Int8 => {
                let DenseGateUpWeights::Int8 { gate, up } = weights.ok_or(
                    Error::InvalidExecutionPlan("INT8 MLP requires prepared gate/up weights"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "INT8 MLP received non-INT8 gate/up weights",
                    ));
                };
                Ok(Self::Int8(Box::new([
                    CompressedInt8Bf16Linear::new(
                        backend,
                        tokens,
                        config.attention.hidden_size,
                        config.intermediate_size,
                        gate.clone(),
                    )?,
                    CompressedInt8Bf16Linear::new(
                        backend,
                        tokens,
                        config.attention.hidden_size,
                        config.intermediate_size,
                        up.clone(),
                    )?,
                ])))
            },
            ProjectionFormat::NvFp4 => {
                let DenseGateUpWeights::NvFp4 { gate, up } = weights.ok_or(
                    Error::InvalidExecutionPlan("NVFP4 MLP requires prepared gate/up weights"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 MLP received non-NVFP4 gate/up weights",
                    ));
                };
                Ok(Self::NvFp4(Box::new(NvFp4Bf16Pack::from_weights(
                    backend,
                    tokens,
                    [gate.clone(), up.clone()],
                )?)))
            },
        }
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        input_norm: &RmsNormBf16,
        norm_weight: &CudaTensor,
        weights: DenseGateUpWeights<'_>,
        buffers: &mut GateUpBuffers<'_>,
    ) -> Result<bool> {
        match self {
            Self::Bf16(operation) => {
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                operation.execute(buffers.normalized, weights.require_bf16()?, buffers.packed)?;
                Ok(false)
            },
            Self::Int8(operations) => {
                let DenseGateUpWeights::Int8 { .. } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "INT8 MLP operation received non-INT8 weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for (operation, output) in operations.iter().zip(buffers.separate.iter_mut()) {
                    operation.execute(buffers.normalized, output)?;
                }
                Ok(true)
            },
            Self::NvFp4(operation) => {
                let DenseGateUpWeights::NvFp4 { .. } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 MLP received BF16 gate/up weights",
                    ));
                };
                operation.execute_rms_norm(
                    input,
                    input_norm.weight(norm_weight)?,
                    input_norm.epsilon(),
                    buffers.separate,
                )?;
                Ok(true)
            },
        }
    }
}

impl DownProjection {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        tokens: usize,
        weight: Option<DenseDownWeight<'_>>,
    ) -> Result<Self> {
        match config.attention.projection_format {
            ProjectionFormat::Bf16 => Ok(Self::Bf16(Bf16Linear::new(
                backend,
                tokens,
                config.intermediate_size,
                config.attention.hidden_size,
            )?)),
            ProjectionFormat::Int8 => {
                let DenseDownWeight::Int8(weight) = weight
                    .ok_or(Error::InvalidExecutionPlan("INT8 MLP requires prepared down weight"))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "INT8 MLP received non-INT8 down weight",
                    ));
                };
                Ok(Self::Int8(CompressedInt8Bf16Linear::new(
                    backend,
                    tokens,
                    config.intermediate_size,
                    config.attention.hidden_size,
                    weight.clone(),
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

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: DenseDownWeight<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Bf16(operation) => operation.execute(input, weight.require_bf16()?, output),
            Self::Int8(operation) => match weight {
                DenseDownWeight::Int8(_) => operation.execute(input, output),
                _ => {
                    Err(Error::InvalidExecutionPlan("INT8 down operation received non-INT8 weight"))
                },
            },
            Self::NvFp4(operation) => match weight {
                DenseDownWeight::NvFp4(_) => operation.execute(input, output),
                DenseDownWeight::Bf16(_) | DenseDownWeight::Int8(_) => {
                    Err(Error::InvalidExecutionPlan("NVFP4 down operation received BF16 weight"))
                },
            },
        }
    }

    pub(super) fn execute_gated(
        &mut self,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        activation: crate::GatedActivation,
        weight: DenseDownWeight<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<bool> {
        match (self, weight) {
            (Self::Bf16(_), DenseDownWeight::Bf16(_))
            | (Self::Int8(_), DenseDownWeight::Int8(_)) => Ok(false),
            (Self::NvFp4(operation), DenseDownWeight::NvFp4(_)) => {
                operation.execute_gated(gate, up, activation.into(), output)?;
                Ok(true)
            },
            (Self::Bf16(_), DenseDownWeight::Int8(_) | DenseDownWeight::NvFp4(_)) => {
                Err(Error::InvalidExecutionPlan("BF16 down operation received NVFP4 weight"))
            },
            (Self::Int8(_), DenseDownWeight::Bf16(_) | DenseDownWeight::NvFp4(_)) => {
                Err(Error::InvalidExecutionPlan("INT8 down operation received other weight"))
            },
            (Self::NvFp4(_), DenseDownWeight::Bf16(_) | DenseDownWeight::Int8(_)) => {
                Err(Error::InvalidExecutionPlan("NVFP4 down operation received BF16 weight"))
            },
        }
    }
}
