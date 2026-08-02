use mircuda::{DeviceBuffer, Stream, bf16};

use super::super::{CudaBackend, DenseGateUpWeights, DenseSwiGluConfig};
use crate::{
    Bf16LinearPair, CudaTensor, DenseExecution, DensePlanRequest, DenseRole, DirectFp8Bf16Linear,
    Error, ExecutionPhase, MxFp4Bf16Linear, MxFp8Bf16Linear, NvFp4Bf16Pack,
    PackedIntegerBf16Linear, ProjectionFormat, Result, RmsNormBf16,
    backend::linear::AffineProjection,
};
pub(in crate::backend::dense) enum GateUpProjection {
    Affine(Box<[AffineProjection; 2]>),
    Bf16(Bf16LinearPair),
    DirectFp8(Box<[DirectFp8Bf16Linear; 2]>),
    MxFp4(Box<[MxFp4Bf16Linear; 2]>),
    MxFp8(Box<[MxFp8Bf16Linear; 2]>),
    PackedInteger(Box<[PackedIntegerBf16Linear; 2]>),
    NvFp4(Box<NvFp4Bf16Pack<2>>),
    BlockFp8,
    Fp8Int4,
}
pub(in crate::backend::dense) struct GateUpBuffers<'a> {
    pub(in crate::backend::dense) normalized: &'a mut DeviceBuffer<bf16>,
    pub(in crate::backend::dense) packed: &'a mut DeviceBuffer<bf16>,
    pub(in crate::backend::dense) separate: &'a mut [DeviceBuffer<bf16>; 2],
}
impl GateUpProjection {
    #[allow(clippy::too_many_lines)]
    pub(in crate::backend::dense) fn new(
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
            ProjectionFormat::Affine => {
                let DenseGateUpWeights::Affine { gate, up } = weights.ok_or(
                    Error::InvalidExecutionPlan("affine MLP requires prepared gate/up weights"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine MLP received non-affine gate/up weights",
                    ));
                };
                Ok(Self::Affine(Box::new([
                    super::affine::gate_up(backend, tokens, config, gate)?,
                    super::affine::gate_up(backend, tokens, config, up)?,
                ])))
            },
            ProjectionFormat::Bf16 => {
                let packed_outputs = config
                    .intermediate_size
                    .checked_mul(2)
                    .ok_or(Error::InvalidDecoderKernel("paired BF16 output size overflow"))?;
                let request = DensePlanRequest {
                    phase,
                    role: DenseRole::DenseGateUp,
                    tokens,
                    input_features: config.attention.hidden_size,
                    output_features: packed_outputs,
                };
                Ok(
                    match backend
                        .execution_planner()
                        .plan_dense_with_prepared_weights(request)?
                        .execution()
                    {
                        DenseExecution::BlockFp8Vector => Self::BlockFp8,
                        DenseExecution::Fp8Int4Vector => Self::Fp8Int4,
                        DenseExecution::Matrix
                        | DenseExecution::Vector
                        | DenseExecution::CublasLt => Self::Bf16(Bf16LinearPair::new(
                            backend,
                            phase,
                            tokens,
                            config.attention.hidden_size,
                            config.intermediate_size,
                        )?),
                    },
                )
            },
            ProjectionFormat::DirectFp8 => {
                let DenseGateUpWeights::DirectFp8 { gate, up } = weights.ok_or(
                    Error::InvalidExecutionPlan("direct FP8 MLP requires gate/up weights"),
                )?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 MLP received other gate/up weights",
                    ));
                };
                Ok(Self::DirectFp8(Box::new([
                    gate.prepare(backend, tokens)?,
                    up.prepare(backend, tokens)?,
                ])))
            },
            ProjectionFormat::MxFp8 => {
                Ok(Self::MxFp8(super::mxfp8::prepare_gate_up(backend, tokens, weights)?))
            },
            ProjectionFormat::MxFp4 => {
                Ok(Self::MxFp4(super::mxfp4::prepare_gate_up(backend, tokens, weights)?))
            },
            ProjectionFormat::PackedInteger => {
                let DenseGateUpWeights::PackedInteger { gate, up } =
                    weights.ok_or(Error::InvalidExecutionPlan(
                        "packed integer MLP requires prepared gate/up weights",
                    ))?
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer MLP received other gate/up weights",
                    ));
                };
                Ok(Self::PackedInteger(Box::new([
                    PackedIntegerBf16Linear::new(
                        backend,
                        tokens,
                        config.attention.hidden_size,
                        config.intermediate_size,
                        gate,
                    )?,
                    PackedIntegerBf16Linear::new(
                        backend,
                        tokens,
                        config.attention.hidden_size,
                        config.intermediate_size,
                        up,
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

    pub(in crate::backend::dense) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        input_norm: &RmsNormBf16,
        norm_weight: &CudaTensor,
        weights: DenseGateUpWeights<'_>,
        buffers: &mut GateUpBuffers<'_>,
    ) -> Result<bool> {
        match self {
            Self::Affine(operations) => {
                let DenseGateUpWeights::Affine { gate, up } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "affine gate/up operation received non-affine weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip([gate, up]).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
            },
            Self::Bf16(operation) => {
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                operation.execute(buffers.normalized, weights.require_bf16()?, buffers.packed)?;
                Ok(false)
            },
            Self::DirectFp8(operations) => {
                let DenseGateUpWeights::DirectFp8 { gate, up } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 gate/up operation received other weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip([gate, up]).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
                }
                Ok(true)
            },
            Self::MxFp8(operations) => super::mxfp8::execute_gate_up(
                operations, input, input_norm, norm_weight, weights, buffers,
            ),
            Self::MxFp4(operations) => super::mxfp4::execute_gate_up(
                operations, input, input_norm, norm_weight, weights, buffers,
            ),
            Self::PackedInteger(operations) => {
                let DenseGateUpWeights::PackedInteger { gate, up } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer MLP operation received other weights",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                for ((operation, weight), output) in
                    operations.iter().zip([gate, up]).zip(buffers.separate.iter_mut())
                {
                    operation.execute(buffers.normalized, weight, output)?;
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
            Self::BlockFp8 => {
                let DenseGateUpWeights::BlockFp8 { quantized, .. } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "block FP8 gate/up plan lacks quantized weight",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                quantized.execute(stream, buffers.normalized, buffers.packed)?;
                Ok(false)
            },
            Self::Fp8Int4 => {
                let DenseGateUpWeights::Fp8Int4 { quantized, .. } = weights else {
                    return Err(Error::InvalidExecutionPlan(
                        "FP8 plus INT4 gate/up plan lacks quantized weight",
                    ));
                };
                input_norm.execute(input, norm_weight, buffers.normalized)?;
                quantized.execute(stream, buffers.normalized, buffers.packed)?;
                Ok(false)
            },
        }
    }
}
