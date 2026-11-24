use mircuda::{DeviceBuffer, Stream, bf16};

use super::{AttentionOutputProjection, DecodeAttentionOutputWeight};
use crate::{Error, Result};

impl AttentionOutputProjection {
    pub(in crate::backend::attention) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: DecodeAttentionOutputWeight<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weight) {
            (Self::Affine(operation), DecodeAttentionOutputWeight::Affine(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Affine(_), _) => {
                Err(Error::InvalidExecutionPlan("affine attention output received other weight"))
            },
            (
                Self::PackedInteger(operation),
                DecodeAttentionOutputWeight::PackedInteger(weight),
            ) => operation.execute(input, weight, output),
            (Self::PackedInteger(_), _) => Err(Error::InvalidExecutionPlan(
                "packed integer attention output received other weight",
            )),
            (Self::NvFp4(operation), DecodeAttentionOutputWeight::NvFp4(_)) => {
                operation.execute(input, output)
            },
            (Self::NvFp4(_), _) => {
                Err(Error::InvalidExecutionPlan("NVFP4 attention output received non-NVFP4 weight"))
            },
            (Self::Bf16(operation), weight) => operation.execute(
                input,
                weight
                    .bf16()
                    .ok_or(Error::InvalidExecutionPlan("BF16 attention lacks BF16 weight"))?,
                output,
            ),
            (Self::DirectFp8(operation), DecodeAttentionOutputWeight::DirectFp8(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::DirectFp8(_), _) => Err(Error::InvalidExecutionPlan(
                "direct FP8 attention output received other weight",
            )),
            (Self::MxFp4(operation), DecodeAttentionOutputWeight::MxFp4(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Ggml(operation), DecodeAttentionOutputWeight::Ggml(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::MxFp4(_), _) => {
                Err(Error::InvalidExecutionPlan("MXFP4 attention output received other weight"))
            },
            (Self::Ggml(_), _) => {
                Err(Error::InvalidExecutionPlan("GGML attention output received other weight"))
            },
            (Self::MxFp8(operation), DecodeAttentionOutputWeight::MxFp8(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::MxFp8(_), _) => {
                Err(Error::InvalidExecutionPlan("MXFP8 attention output received other weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::BlockFp8 { quantized, .. }) => {
                quantized.execute(stream, input, output)
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::Bf16(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan lacks quantized weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::DirectFp8(_)) => Err(
                Error::InvalidExecutionPlan("block FP8 attention plan received direct FP8 weight"),
            ),
            (Self::BlockFp8, DecodeAttentionOutputWeight::MxFp8(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan received MXFP8 weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::MxFp4(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan received MXFP4 weight"))
            },
            (Self::BlockFp8, DecodeAttentionOutputWeight::Ggml(_)) => {
                Err(Error::InvalidExecutionPlan("block FP8 attention plan received GGML weight"))
            },
            (Self::Fp8Int4, DecodeAttentionOutputWeight::Fp8Int4 { quantized, .. }) => {
                quantized.execute(stream, input, output)
            },
            (Self::Fp8Int4, _) => Err(Error::InvalidExecutionPlan(
                "FP8 plus INT4 attention plan lacks quantized weight",
            )),
            (Self::BlockFp8, DecodeAttentionOutputWeight::Fp8Int4 { .. }) => Err(
                Error::InvalidExecutionPlan("block FP8 attention plan received residual weight"),
            ),
            (_, DecodeAttentionOutputWeight::NvFp4(_)) => {
                Err(Error::InvalidExecutionPlan("NVFP4 attention output execution is not prepared"))
            },
            (_, DecodeAttentionOutputWeight::PackedInteger(_)) => Err(Error::InvalidExecutionPlan(
                "packed integer attention output execution is not prepared",
            )),
            (_, DecodeAttentionOutputWeight::Affine(_)) => Err(Error::InvalidExecutionPlan(
                "affine attention output execution is not prepared",
            )),
        }
    }
}
