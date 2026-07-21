use mircuda::{DeviceBuffer, Stream, bf16, cuda_export};

use super::{ClampedRoutedKernels, linear_launch, narrow};
use crate::Result;

cuda_export!(pub(super) QkvKernel = "libmir_cuda_clamped_routed_qkv_bf16"(
    packed: &DeviceBuffer<bf16>, q_bias: &DeviceBuffer<bf16>,
    k_bias: &DeviceBuffer<bf16>, v_bias: &DeviceBuffer<bf16>,
    query: &mut DeviceBuffer<bf16>, key: &mut DeviceBuffer<bf16>,
    value: &mut DeviceBuffer<bf16>, tokens: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, start_position: u32, theta: f32,
    factor: f32, initial_context: f32, beta_fast: f32, beta_slow: f32,
));
cuda_export!(pub(super) QkvSplitKernel = "libmir_cuda_clamped_routed_qkv_split_bf16"(
    q_input: &DeviceBuffer<bf16>, k_input: &DeviceBuffer<bf16>,
    v_input: &DeviceBuffer<bf16>, q_bias: &DeviceBuffer<bf16>,
    k_bias: &DeviceBuffer<bf16>, v_bias: &DeviceBuffer<bf16>,
    query: &mut DeviceBuffer<bf16>, key: &mut DeviceBuffer<bf16>,
    value: &mut DeviceBuffer<bf16>, tokens: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, start_position: u32, theta: f32,
    factor: f32, initial_context: f32, beta_fast: f32, beta_slow: f32,
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
        start: usize,
    ) -> Result<()> {
        let biases = bias_buffers(biases)?;
        let width = self.spec.query_heads * self.spec.head_dim
            + 2 * self.spec.kv_heads * self.spec.head_dim;
        Ok(self.qkv.launch(
            stream,
            linear_launch(self.spec.tokens * width)?,
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
                narrow(start)?,
                self.spec.theta,
                self.spec.factor,
                self.spec.initial_context,
                self.spec.beta_fast,
                self.spec.beta_slow,
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
        start: usize,
    ) -> Result<()> {
        let biases = bias_buffers(biases)?;
        let width = self.spec.query_heads * self.spec.head_dim
            + 2 * self.spec.kv_heads * self.spec.head_dim;
        Ok(self.qkv_split.launch(
            stream,
            linear_launch(self.spec.tokens * width)?,
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
                narrow(start)?,
                self.spec.theta,
                self.spec.factor,
                self.spec.initial_context,
                self.spec.beta_fast,
                self.spec.beta_slow,
            ),
        )?)
    }
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
