use mircuda::DeviceBuffer;

use super::SpatialMergeLayer;
use crate::{
    Result,
    backend::vision::spatial_merge::{primitives::bf16, scratch::SpatialMergeScratch},
};

impl SpatialMergeLayer {
    pub(in crate::backend::vision::spatial_merge) fn execute(
        &mut self,
        scratch: &mut SpatialMergeScratch,
        positions: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.elementwise_hidden.layer_norm(
            stream,
            &scratch.hidden_a,
            bf16(&self.norm1_weight)?,
            bf16(&self.norm1_bias)?,
            &mut scratch.normalized,
        )?;
        self.qkv.execute(&scratch.normalized, &mut scratch.qkv)?;
        self.kernels.split_qkv(
            stream,
            &scratch.qkv,
            &mut scratch.query,
            &mut scratch.key,
            &mut scratch.value,
            self.tokens,
            self.heads * self.head_dim,
        )?;
        self.kernels.rope(
            stream,
            &scratch.query,
            positions,
            &mut scratch.query_rope,
            self.tokens,
            self.heads,
            self.head_dim,
        )?;
        self.kernels.rope(
            stream,
            &scratch.key,
            positions,
            &mut scratch.key_rope,
            self.tokens,
            self.heads,
            self.head_dim,
        )?;
        self.attention.execute(
            stream,
            &scratch.query_rope,
            &scratch.key_rope,
            &scratch.value,
            &mut scratch.hidden_b,
        )?;
        self.projection.execute(&scratch.hidden_b, &mut scratch.normalized)?;
        self.elementwise_hidden.add(
            stream,
            &scratch.hidden_a,
            &scratch.normalized,
            &mut scratch.hidden_b,
        )?;
        self.elementwise_hidden.layer_norm(
            stream,
            &scratch.hidden_b,
            bf16(&self.norm2_weight)?,
            bf16(&self.norm2_bias)?,
            &mut scratch.normalized,
        )?;
        self.fc1.execute(&scratch.normalized, &mut scratch.intermediate_a)?;
        self.elementwise_intermediate.gelu(
            stream,
            &scratch.intermediate_a,
            &mut scratch.intermediate_b,
            true,
        )?;
        self.fc2.execute(&scratch.intermediate_b, &mut scratch.hidden_a)?;
        self.elementwise_hidden.add(
            stream,
            &scratch.hidden_b,
            &scratch.hidden_a,
            &mut scratch.normalized,
        )?;
        std::mem::swap(&mut scratch.hidden_a, &mut scratch.normalized);
        Ok(())
    }
}
