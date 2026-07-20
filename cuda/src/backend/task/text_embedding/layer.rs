use mircuda::{DeviceBuffer, bf16};

use super::{
    CudaTextEmbeddingModel,
    execute::{epsilon, project, rope, theta},
    scratch::Scratch,
};
use crate::{
    Result,
    kernels::{TextAttention, TextAttentionSpec},
};

impl CudaTextEmbeddingModel {
    #[expect(clippy::too_many_lines, reason = "keeps one decoder layer contract contiguous")]
    pub(super) fn layer(
        &self,
        layer: usize,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        scratch: &mut Scratch,
    ) -> Result<()> {
        let config = &self.config;
        let count = scratch.tokens;
        let hidden = config.hidden_size;
        let prefix = self.layout.name(format!("layers.{layer}"));
        let backend = &self.backend;
        backend.prepare_rms_norm_bf16(count, hidden, epsilon(config)?)?.execute(
            input,
            self.tensor(&format!("{prefix}.input_layernorm.weight"))?,
            &mut scratch.normalized,
        )?;
        project(
            backend,
            count,
            hidden,
            scratch.query_width,
            &scratch.normalized,
            self.tensor(&format!("{prefix}.self_attn.q_proj.weight"))?,
            &mut scratch.query,
        )?;
        project(
            backend,
            count,
            hidden,
            scratch.kv_width,
            &scratch.normalized,
            self.tensor(&format!("{prefix}.self_attn.k_proj.weight"))?,
            &mut scratch.key,
        )?;
        project(
            backend,
            count,
            hidden,
            scratch.kv_width,
            &scratch.normalized,
            self.tensor(&format!("{prefix}.self_attn.v_proj.weight"))?,
            &mut scratch.value,
        )?;
        let head = config.head_dim;
        backend
            .prepare_rms_norm_bf16(count * config.num_attention_heads, head, epsilon(config)?)?
            .execute(
                &scratch.query,
                self.tensor(&format!("{prefix}.self_attn.q_norm.weight"))?,
                &mut scratch.query_norm,
            )?;
        backend
            .prepare_rms_norm_bf16(count * config.num_key_value_heads, head, epsilon(config)?)?
            .execute(
                &scratch.key,
                self.tensor(&format!("{prefix}.self_attn.k_norm.weight"))?,
                &mut scratch.key_norm,
            )?;
        rope(backend, count, config.num_attention_heads, head, theta(config)?)?.execute(
            &scratch.query_norm,
            &mut scratch.query_rope,
            0,
        )?;
        rope(backend, count, config.num_key_value_heads, head, theta(config)?)?.execute(
            &scratch.key_norm,
            &mut scratch.key_rope,
            0,
        )?;
        TextAttention::compile(
            &backend.inner.compiler,
            TextAttentionSpec {
                tokens: count,
                query_heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: head,
                scale: head.to_string().parse::<f32>()?.sqrt().recip(),
                causal: true,
            },
        )?
        .execute(
            &backend.inner.stream,
            &scratch.query_rope,
            &scratch.key_rope,
            &scratch.value,
            &mut scratch.attention,
        )?;
        project(
            backend,
            count,
            scratch.query_width,
            hidden,
            &scratch.attention,
            self.tensor(&format!("{prefix}.self_attn.o_proj.weight"))?,
            &mut scratch.projection,
        )?;
        scratch.hidden_ops.add(
            &backend.inner.stream,
            input,
            &scratch.projection,
            &mut scratch.residual,
        )?;
        backend.prepare_rms_norm_bf16(count, hidden, epsilon(config)?)?.execute(
            &scratch.residual,
            self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            &mut scratch.normalized,
        )?;
        project(
            backend,
            count,
            hidden,
            config.intermediate_size,
            &scratch.normalized,
            self.tensor(&format!("{prefix}.mlp.gate_proj.weight"))?,
            &mut scratch.gate,
        )?;
        project(
            backend,
            count,
            hidden,
            config.intermediate_size,
            &scratch.normalized,
            self.tensor(&format!("{prefix}.mlp.up_proj.weight"))?,
            &mut scratch.up,
        )?;
        scratch.gated.execute_separate(
            &backend.inner.stream,
            &scratch.gate,
            &scratch.up,
            &mut scratch.activated,
            crate::GatedActivation::Silu.into(),
        )?;
        project(
            backend,
            count,
            config.intermediate_size,
            hidden,
            &scratch.activated,
            self.tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
            &mut scratch.mlp,
        )?;
        scratch
            .hidden_ops
            .add(&backend.inner.stream, &scratch.residual, &scratch.mlp, output)
    }
}
