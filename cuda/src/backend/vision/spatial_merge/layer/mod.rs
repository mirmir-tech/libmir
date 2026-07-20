mod execution;

use models::layout::SpatialMergeVisionConfig;

use super::super::{
    super::super::CudaBackend,
    linear::{VisionLinear, required},
};
use crate::{
    CudaTensor, CudaTensorSet, Result,
    kernels::{
        SpatialMergeKernels, VisionAttention, VisionAttentionSpec, VisionElementwise,
        VisionElementwiseSpec,
    },
};

#[derive(Debug)]
pub(super) struct SpatialMergeLayer {
    backend: CudaBackend,
    norm1_weight: CudaTensor,
    norm1_bias: CudaTensor,
    norm2_weight: CudaTensor,
    norm2_bias: CudaTensor,
    qkv: VisionLinear,
    projection: VisionLinear,
    fc1: VisionLinear,
    fc2: VisionLinear,
    elementwise_hidden: VisionElementwise,
    elementwise_intermediate: VisionElementwise,
    attention: VisionAttention,
    kernels: SpatialMergeKernels,
    tokens: usize,
    heads: usize,
    head_dim: usize,
}

impl SpatialMergeLayer {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &SpatialMergeVisionConfig,
        tensors: &CudaTensorSet,
        prefix: &str,
        index: usize,
        tokens: usize,
        kernels: SpatialMergeKernels,
    ) -> Result<Self> {
        let layer = format!("{prefix}.blocks.{index}");
        let linear = |name: &str, input, output| {
            VisionLinear::new(backend, tensors, name, tokens, input, output, false)
        };
        let compiler = &backend.inner.compiler;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let scale = 1.0 / head_dim.to_string().parse::<f32>()?.sqrt();
        Ok(Self {
            backend: backend.clone(),
            norm1_weight: required(tensors, &format!("{layer}.norm1.weight"))?,
            norm1_bias: required(tensors, &format!("{layer}.norm1.bias"))?,
            norm2_weight: required(tensors, &format!("{layer}.norm2.weight"))?,
            norm2_bias: required(tensors, &format!("{layer}.norm2.bias"))?,
            qkv: linear(&format!("{layer}.attn.qkv"), config.hidden_size, 3 * config.hidden_size)?,
            projection: linear(
                &format!("{layer}.attn.proj"),
                config.hidden_size,
                config.hidden_size,
            )?,
            fc1: linear(
                &format!("{layer}.mlp.linear_fc1"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            fc2: linear(
                &format!("{layer}.mlp.linear_fc2"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            elementwise_hidden: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: tokens,
                    columns: config.hidden_size,
                    epsilon: 1.0e-6,
                },
            )?,
            elementwise_intermediate: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: tokens,
                    columns: config.intermediate_size,
                    epsilon: 0.0,
                },
            )?,
            attention: VisionAttention::compile(
                compiler,
                VisionAttentionSpec {
                    tokens,
                    query_heads: config.num_attention_heads,
                    kv_heads: config.num_attention_heads,
                    head_dim,
                    scale,
                },
            )?,
            kernels,
            tokens,
            heads: config.num_attention_heads,
            head_dim,
        })
    }
}
