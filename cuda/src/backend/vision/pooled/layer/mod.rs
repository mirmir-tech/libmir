use models::layout::PooledVisionConfig;

mod execution;

use super::super::{
    super::CudaBackend,
    linear::{VisionLinear, required},
};
use crate::{
    CudaTensor, CudaTensorSet, Result,
    kernels::{
        RmsNorm, RmsNormUnit, VisionAttention, VisionAttentionSpec, VisionElementwise,
        VisionElementwiseSpec, VisionSpatialRope,
    },
};

#[derive(Debug)]
pub(super) struct PooledLayer {
    backend: CudaBackend,
    input_norm: RmsNorm,
    input_weight: CudaTensor,
    query_norm: RmsNorm,
    query_weight: CudaTensor,
    key_norm: RmsNorm,
    key_weight: CudaTensor,
    value_norm: RmsNormUnit,
    query: VisionLinear,
    key: VisionLinear,
    value: VisionLinear,
    output: VisionLinear,
    rope_query: VisionSpatialRope,
    rope_key: VisionSpatialRope,
    attention: VisionAttention,
    post_attention_norm: RmsNorm,
    post_attention_weight: CudaTensor,
    pre_feedforward_norm: RmsNorm,
    pre_feedforward_weight: CudaTensor,
    gate: VisionLinear,
    up: VisionLinear,
    down: VisionLinear,
    post_feedforward_norm: RmsNorm,
    post_feedforward_weight: CudaTensor,
    elementwise_hidden: VisionElementwise,
    elementwise_intermediate: VisionElementwise,
}

impl PooledLayer {
    #[allow(clippy::too_many_lines)]
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PooledVisionConfig,
        tensors: &CudaTensorSet,
        index: usize,
        tokens: usize,
    ) -> Result<Self> {
        let prefix = format!("model.vision_tower.encoder.layers.{index}");
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        let eps = config.rms_norm_eps.to_string().parse::<f32>()?;
        let compiler = &backend.inner.compiler;
        let rms = || RmsNorm::compile(compiler, tokens, config.hidden_size, eps);
        let linear = |prefix: &str, input, output| {
            VisionLinear::new(
                backend,
                tensors,
                prefix,
                tokens,
                input,
                output,
                config.use_clipped_linears,
            )
        };
        Ok(Self {
            backend: backend.clone(),
            input_norm: rms()?,
            input_weight: required(tensors, &format!("{prefix}.input_layernorm.weight"))?,
            query_norm: RmsNorm::compile(
                compiler,
                tokens * config.num_attention_heads,
                config.head_dim,
                eps,
            )?,
            query_weight: required(tensors, &format!("{attention}.q_norm.weight"))?,
            key_norm: RmsNorm::compile(
                compiler,
                tokens * config.num_key_value_heads,
                config.head_dim,
                eps,
            )?,
            key_weight: required(tensors, &format!("{attention}.k_norm.weight"))?,
            value_norm: RmsNormUnit::compile(
                compiler,
                tokens * config.num_key_value_heads,
                config.head_dim,
                eps,
            )?,
            query: linear(&format!("{attention}.q_proj"), config.hidden_size, config.hidden_size)?,
            key: linear(
                &format!("{attention}.k_proj"),
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            value: linear(
                &format!("{attention}.v_proj"),
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            output: linear(&format!("{attention}.o_proj"), config.hidden_size, config.hidden_size)?,
            rope_query: VisionSpatialRope::compile(
                compiler,
                tokens,
                config.num_attention_heads,
                config.head_dim,
                config.rope_theta.to_string().parse::<f32>()?,
            )?,
            rope_key: VisionSpatialRope::compile(
                compiler,
                tokens,
                config.num_key_value_heads,
                config.head_dim,
                config.rope_theta.to_string().parse::<f32>()?,
            )?,
            attention: VisionAttention::compile(
                compiler,
                VisionAttentionSpec {
                    tokens,
                    query_heads: config.num_attention_heads,
                    kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    scale: 1.0,
                },
            )?,
            post_attention_norm: rms()?,
            post_attention_weight: required(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            pre_feedforward_norm: rms()?,
            pre_feedforward_weight: required(
                tensors,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
            )?,
            gate: linear(
                &format!("{mlp}.gate_proj"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            up: linear(&format!("{mlp}.up_proj"), config.hidden_size, config.intermediate_size)?,
            down: linear(
                &format!("{mlp}.down_proj"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            post_feedforward_norm: rms()?,
            post_feedforward_weight: required(
                tensors,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
            )?,
            elementwise_hidden: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: tokens,
                    columns: config.hidden_size,
                    epsilon: eps,
                },
            )?,
            elementwise_intermediate: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: tokens,
                    columns: config.intermediate_size,
                    epsilon: eps,
                },
            )?,
        })
    }
}
