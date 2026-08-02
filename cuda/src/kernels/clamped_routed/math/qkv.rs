use mircuda::{DeviceBuffer, Stream, bf16, cuda_export};

use super::{ClampedRoutedKernels, linear_launch, narrow};
use crate::Result;

cuda_export!(pub(super) QkvKernel = "libmir_cuda_clamped_routed_qkv_bf16"(
    packed: &DeviceBuffer<bf16>, q_bias: &DeviceBuffer<bf16>,
    k_bias: &DeviceBuffer<bf16>, v_bias: &DeviceBuffer<bf16>,
    query: &mut DeviceBuffer<bf16>, key: &mut DeviceBuffer<bf16>,
    value: &mut DeviceBuffer<bf16>, tokens: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, rope_sines: &DeviceBuffer<f32>,
    rope_cosines: &DeviceBuffer<f32>, concentration: f32,
));
cuda_export!(pub(super) QkvSplitKernel = "libmir_cuda_clamped_routed_qkv_split_bf16"(
    q_input: &DeviceBuffer<bf16>, k_input: &DeviceBuffer<bf16>,
    v_input: &DeviceBuffer<bf16>, q_bias: &DeviceBuffer<bf16>,
    k_bias: &DeviceBuffer<bf16>, v_bias: &DeviceBuffer<bf16>,
    query: &mut DeviceBuffer<bf16>, key: &mut DeviceBuffer<bf16>,
    value: &mut DeviceBuffer<bf16>, tokens: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, rope_sines: &DeviceBuffer<f32>,
    rope_cosines: &DeviceBuffer<f32>, concentration: f32,
));
cuda_export!(pub(super) RopeKernel = "libmir_cuda_clamped_routed_rope_angles"(
    positions: &DeviceBuffer<u32>, inverse_frequencies: &DeviceBuffer<f32>,
    sines: &mut DeviceBuffer<f32>, cosines: &mut DeviceBuffer<f32>,
    tokens: u32, half_head_dim: u32,
));

impl ClampedRoutedKernels {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qkv_native(
        &self,
        stream: &Stream,
        packed: &DeviceBuffer<bf16>,
        biases: &[crate::CudaTensor; 3],
        query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>,
        value: &mut DeviceBuffer<bf16>,
        rope_sines: &DeviceBuffer<f32>,
        rope_cosines: &DeviceBuffer<f32>,
        concentration: f32,
    ) -> Result<()> {
        let biases = bias_buffers(biases)?;
        let elements = qkv_elements(self.spec);
        Ok(self.qkv.launch(
            stream,
            linear_launch(elements)?,
            (
                packed,
                biases[0],
                biases[1],
                biases[2],
                query,
                key,
                value,
                narrow(self.spec.tokens)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                rope_sines,
                rope_cosines,
                concentration,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qkv_mlx(
        &self,
        stream: &Stream,
        inputs: [&DeviceBuffer<bf16>; 3],
        biases: &[crate::CudaTensor; 3],
        query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>,
        value: &mut DeviceBuffer<bf16>,
        rope_sines: &DeviceBuffer<f32>,
        rope_cosines: &DeviceBuffer<f32>,
        concentration: f32,
    ) -> Result<()> {
        let biases = bias_buffers(biases)?;
        let elements = qkv_elements(self.spec);
        Ok(self.qkv_split.launch(
            stream,
            linear_launch(elements)?,
            (
                inputs[0],
                inputs[1],
                inputs[2],
                biases[0],
                biases[1],
                biases[2],
                query,
                key,
                value,
                narrow(self.spec.tokens)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                rope_sines,
                rope_cosines,
                concentration,
            ),
        )?)
    }

    pub(crate) fn prepare_rope(
        &self,
        stream: &Stream,
        positions: &DeviceBuffer<u32>,
        inverse: &DeviceBuffer<f32>,
        sines: &mut DeviceBuffer<f32>,
        cosines: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let half = self.spec.head_dim / 2;
        let elements = self.spec.tokens * half;
        Ok(self.rope.launch(
            stream,
            linear_launch(elements)?,
            (positions, inverse, sines, cosines, narrow(self.spec.tokens)?, narrow(half)?),
        )?)
    }
}

const fn qkv_elements(spec: super::ClampedRoutedSpec) -> usize {
    let rotary = (spec.query_heads + spec.kv_heads) * (spec.head_dim / 2);
    let values = spec.kv_heads * spec.head_dim;
    spec.tokens * (rotary + values)
}

fn bias_buffers(biases: &[crate::CudaTensor; 3]) -> Result<[&DeviceBuffer<bf16>; 3]> {
    Ok([bf16_tensor(&biases[0])?, bf16_tensor(&biases[1])?, bf16_tensor(&biases[2])?])
}

fn bf16_tensor(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| crate::Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
