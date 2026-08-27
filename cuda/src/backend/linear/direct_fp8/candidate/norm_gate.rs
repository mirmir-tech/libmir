use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    Candidate, DirectFp8Activation, DirectFp8CheckpointWeight, DirectFp8Format, Operation,
};
use crate::{Error, Result, backend::linear::direct_fp8::candidate::bias};

impl Candidate {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::linear::direct_fp8) fn execute_norm_gate(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        norm: &DeviceBuffer<bf16>,
        weight: &DirectFp8CheckpointWeight,
        epsilon: f32,
        weight_shift: f32,
        value_heads: usize,
        gate_stride: usize,
        gate_offset: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<bool> {
        let Operation::TensorCore(operation) = &self.operation else {
            return Ok(false);
        };
        if weight.activation != DirectFp8Activation::DynamicE4M3Token
            || weight.format != DirectFp8Format::E4M3
        {
            return Ok(false);
        }
        let weight_buffer = weight.weight.as_f8_e4m3().ok_or_else(|| Error::DTypeMismatch {
            name: weight.weight.name().into(),
            expected: "F8_E4M3",
        })?;
        if let Some(weight_scales) = weight.scales.as_ref().and_then(super::CudaTensor::as_f32) {
            operation.execute_norm_gate_f32_scales(
                stream,
                input,
                gate,
                norm,
                weight_buffer,
                weight_scales,
                bias(weight)?,
                epsilon,
                weight_shift,
                value_heads,
                gate_stride,
                gate_offset,
                output,
            )?;
        } else if let Some(weight_scales) =
            weight.scales.as_ref().and_then(super::CudaTensor::as_bf16)
        {
            operation.execute_norm_gate_bf16_scales(
                stream,
                input,
                gate,
                norm,
                weight_buffer,
                weight_scales,
                bias(weight)?,
                epsilon,
                weight_shift,
                value_heads,
                gate_stride,
                gate_offset,
                output,
            )?;
        } else {
            return Err(Error::DTypeMismatch {
                name: weight.weight.name().into(),
                expected: "BF16 or F32 weight scale",
            });
        }
        Ok(true)
    }
}
