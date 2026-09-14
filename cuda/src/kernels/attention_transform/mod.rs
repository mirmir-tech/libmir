use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{
    MropeSpec,
    geometry::{narrow, product, require},
};
use crate::{Error, Result};

#[cfg(all(test, target_os = "linux"))]
mod tests;

cuda_export!(
    AttentionTransformKernel = "libmir_cuda_attention_transform_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>,
        query_weight: &DeviceBuffer<bf16>, key_weight: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>, rotated_query: &mut DeviceBuffer<bf16>,
        rotated_key: &mut DeviceBuffer<bf16>, gate: &mut DeviceBuffer<bf16>,
        tokens: u32, query_heads: u32, key_heads: u32, head_dim: u32, rotary_dim: u32,
        section_t: u32, section_h: u32, section_w: u32, interleaved: u32,
        theta: f32, epsilon: f32, weight_shift: f32,
    )
);

#[derive(Debug)]
pub struct AttentionTransform {
    kernel: TypedKernel<AttentionTransformKernel>,
    spec: MropeSpec,
    key_heads: usize,
    epsilon: f32,
    weight_shift: f32,
}

impl AttentionTransform {
    pub fn compile(
        compiler: &Compiler,
        spec: MropeSpec,
        key_heads: usize,
        epsilon: f32,
        weight_shift: f32,
    ) -> Result<Self> {
        super::mrope::validate(spec)?;
        if key_heads == 0 || !epsilon.is_finite() || epsilon < 0.0 || !weight_shift.is_finite() {
            return Err(Error::InvalidDecoderKernel("invalid attention transform geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/attention_transform_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            spec,
            key_heads,
            epsilon,
            weight_shift,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key: &DeviceBuffer<bf16>,
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        rotated_query: &mut DeviceBuffer<bf16>,
        rotated_key: &mut DeviceBuffer<bf16>,
        gate: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let s = self.spec;
        let queries = product(product(s.tokens, s.heads)?, s.head_dim)?;
        let keys = product(product(s.tokens, self.key_heads)?, s.head_dim)?;
        narrow(product(queries, 2)?)?;
        narrow(keys)?;
        require("attention transform query", product(queries, 2)?, query.len())?;
        require("attention transform key", keys, key.len())?;
        require("attention transform query norm", s.head_dim, query_weight.len())?;
        require("attention transform key norm", s.head_dim, key_weight.len())?;
        require("attention transform positions", product(s.tokens, 3)?, positions.len())?;
        require("attention transform rotated query", queries, rotated_query.len())?;
        require("attention transform rotated key", keys, rotated_key.len())?;
        require("attention transform gate", queries, gate.len())?;
        let heads = s
            .heads
            .checked_add(self.key_heads)
            .ok_or(Error::InvalidDecoderKernel("attention transform heads overflow"))?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(product(s.tokens, heads)?)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key,
                query_weight,
                key_weight,
                positions,
                rotated_query,
                rotated_key,
                gate,
                narrow(s.tokens)?,
                narrow(s.heads)?,
                narrow(self.key_heads)?,
                narrow(s.head_dim)?,
                narrow(s.rotary_dim)?,
                narrow(s.sections[0])?,
                narrow(s.sections[1])?,
                narrow(s.sections[2])?,
                u32::from(s.interleaved),
                s.theta,
                self.epsilon,
                self.weight_shift,
            ),
        )?)
    }
}
