use models::layout::EncoderRopeScaling;

use super::{
    CudaSequenceScoringModel,
    execute::{epsilon, f16_tensor, ops, project},
    scratch::Scratch,
};
use crate::{
    Error, Result,
    kernels::{EncoderAttentionF16, EncoderAttentionSpec},
};

impl CudaSequenceScoringModel {
    #[expect(clippy::too_many_lines, reason = "keeps one encoder layer contract contiguous")]
    pub(super) fn layer(&self, layer: usize, scratch: &mut Scratch) -> Result<()> {
        let config = &self.config;
        let count = scratch.tokens;
        let hidden = config.hidden_size;
        let prefix = format!("new.encoder.layer.{layer}");
        let (input, output) = if layer.is_multiple_of(2) {
            (&scratch.first, &mut scratch.second)
        } else {
            (&scratch.second, &mut scratch.first)
        };
        project(
            &self.backend,
            count,
            hidden,
            hidden * 3,
            input,
            self.tensor(&format!("{prefix}.attention.qkv_proj.weight"))?,
            &mut scratch.qkv_raw,
        )?;
        ops(&self.backend, count, hidden * 3, epsilon(config)?)?.add_bias(
            &self.backend.inner.stream,
            &scratch.qkv_raw,
            f16_tensor(self.tensor(&format!("{prefix}.attention.qkv_proj.bias"))?)?,
            &mut scratch.qkv,
        )?;
        EncoderAttentionF16::compile(
            &self.backend.inner.compiler,
            EncoderAttentionSpec {
                tokens: count,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                theta: config.rope_theta.unwrap_or(10_000.0).to_string().parse()?,
                ntk_factor: ntk_factor(config)?,
            },
        )?
        .execute(&self.backend.inner.stream, &scratch.qkv, &mut scratch.attention)?;
        project(
            &self.backend,
            count,
            hidden,
            hidden,
            &scratch.attention,
            self.tensor(&format!("{prefix}.attention.o_proj.weight"))?,
            &mut scratch.projection_raw,
        )?;
        ops(&self.backend, count, hidden, epsilon(config)?)?.add_bias(
            &self.backend.inner.stream,
            &scratch.projection_raw,
            f16_tensor(self.tensor(&format!("{prefix}.attention.o_proj.bias"))?)?,
            &mut scratch.projection,
        )?;
        ops(&self.backend, count, hidden, epsilon(config)?)?.residual_norm(
            &self.backend.inner.stream,
            input,
            &scratch.projection,
            f16_tensor(self.tensor(&format!("{prefix}.attn_ln.weight"))?)?,
            f16_tensor(self.tensor(&format!("{prefix}.attn_ln.bias"))?)?,
            output,
        )?;
        project(
            &self.backend,
            count,
            hidden,
            config.intermediate_size * 2,
            output,
            self.tensor(&format!("{prefix}.mlp.up_gate_proj.weight"))?,
            &mut scratch.up_gate,
        )?;
        ops(&self.backend, count, config.intermediate_size, epsilon(config)?)?.gated_gelu(
            &self.backend.inner.stream,
            &scratch.up_gate,
            &mut scratch.activated,
        )?;
        project(
            &self.backend,
            count,
            config.intermediate_size,
            hidden,
            &scratch.activated,
            self.tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
            &mut scratch.down_raw,
        )?;
        ops(&self.backend, count, hidden, epsilon(config)?)?.add_bias(
            &self.backend.inner.stream,
            &scratch.down_raw,
            f16_tensor(self.tensor(&format!("{prefix}.mlp.down_proj.bias"))?)?,
            &mut scratch.down,
        )?;
        ops(&self.backend, count, hidden, epsilon(config)?)?.residual_norm(
            &self.backend.inner.stream,
            output,
            &scratch.down,
            f16_tensor(self.tensor(&format!("{prefix}.mlp_ln.weight"))?)?,
            f16_tensor(self.tensor(&format!("{prefix}.mlp_ln.bias"))?)?,
            &mut scratch.projection_raw,
        )?;
        self.backend.inner.stream.copy_device_range(
            &scratch.projection_raw,
            0..count * hidden,
            output,
            0,
        )?;
        Ok(())
    }
}

fn ntk_factor(config: &models::layout::EncoderConfig) -> Result<f32> {
    let Some(EncoderRopeScaling::Ntk { factor, mixed_b: None }) = config.rope_scaling else {
        return Err(Error::InvalidDecoderKernel("encoder fixed NTK factor is missing"));
    };
    Ok(factor.to_string().parse()?)
}
