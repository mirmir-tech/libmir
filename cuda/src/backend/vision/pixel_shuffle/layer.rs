//! One pre-norm transformer block of the encoder: layer norm, fused QKV with
//! bias, bidirectional attention over the prefix and every patch, output
//! projection, residual; layer norm, GELU MLP, residual. No rotary
//! positions: the patches carry absolute positions from the embedding.

use models::layout::PixelShuffleVisionConfig;

use super::{
    super::{
        super::CudaBackend,
        linear::{VisionLinear, required},
    },
    scratch::{PixelShuffleScratch, bf16, single},
};
use crate::{
    CudaTensor, CudaTensorSet, Result,
    kernels::{
        SpatialMergeKernels, VisionAttention, VisionAttentionSpec, VisionElementwise,
        VisionElementwiseSpec,
    },
};

#[derive(Debug)]
pub(super) struct PixelShuffleLayer {
    backend: CudaBackend,
    norms: [CudaTensor; 4],
    qkv: VisionLinear,
    projection: VisionLinear,
    fc1: VisionLinear,
    fc2: VisionLinear,
    hidden: VisionElementwise,
    intermediate: VisionElementwise,
    attention: VisionAttention,
    split: SpatialMergeKernels,
    tokens: usize,
    width: usize,
}

impl PixelShuffleLayer {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PixelShuffleVisionConfig,
        tensors: &CudaTensorSet,
        index: usize,
        tokens: usize,
        split: SpatialMergeKernels,
    ) -> Result<Self> {
        let name = |suffix: &str| {
            models::weights::pixel_shuffle_encoder_tensor(&format!("blocks.{index}.{suffix}"))
        };
        let linear = |suffix: &str, input, output| {
            VisionLinear::new(backend, tensors, &name(suffix), tokens, input, output, false)
        };
        let hidden = config.hidden_size;
        let head_dim = hidden / config.num_attention_heads;
        let compiler = &backend.inner.compiler;
        let epsilon = single(config.layer_norm_eps)?;
        Ok(Self {
            backend: backend.clone(),
            norms: [
                required(tensors, &name("norm1.weight"))?,
                required(tensors, &name("norm1.bias"))?,
                required(tensors, &name("norm2.weight"))?,
                required(tensors, &name("norm2.bias"))?,
            ],
            qkv: linear("attn.qkv", hidden, 3 * hidden)?,
            projection: linear("attn.proj", hidden, hidden)?,
            fc1: linear("mlp.fc1", hidden, config.intermediate_size)?,
            fc2: linear("mlp.fc2", config.intermediate_size, hidden)?,
            hidden: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec { rows: tokens, columns: hidden, epsilon },
            )?,
            intermediate: VisionElementwise::compile(
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
                    scale: head_dim.to_string().parse::<f32>()?.sqrt().recip(),
                },
            )?,
            split,
            tokens,
            width: hidden,
        })
    }

    pub(super) fn execute(&mut self, scratch: &mut PixelShuffleScratch) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let [norm1_weight, norm1_bias, norm2_weight, norm2_bias] = &self.norms;
        self.hidden.layer_norm(
            stream,
            &scratch.hidden_a,
            bf16(norm1_weight)?,
            bf16(norm1_bias)?,
            &mut scratch.normalized,
        )?;
        self.qkv.execute(&scratch.normalized, &mut scratch.qkv)?;
        self.split.split_qkv(
            stream,
            &scratch.qkv,
            &mut scratch.query,
            &mut scratch.key,
            &mut scratch.value,
            self.tokens,
            self.width,
        )?;
        self.attention.execute(
            stream,
            &scratch.query,
            &scratch.key,
            &scratch.value,
            &mut scratch.hidden_b,
        )?;
        self.projection.execute(&scratch.hidden_b, &mut scratch.normalized)?;
        self.hidden
            .add(stream, &scratch.hidden_a, &scratch.normalized, &mut scratch.hidden_b)?;
        self.hidden.layer_norm(
            stream,
            &scratch.hidden_b,
            bf16(norm2_weight)?,
            bf16(norm2_bias)?,
            &mut scratch.normalized,
        )?;
        self.fc1.execute(&scratch.normalized, &mut scratch.intermediate_a)?;
        self.intermediate.gelu(
            stream,
            &scratch.intermediate_a,
            &mut scratch.intermediate_b,
            false,
        )?;
        self.fc2.execute(&scratch.intermediate_b, &mut scratch.hidden_a)?;
        self.hidden
            .add(stream, &scratch.hidden_b, &scratch.hidden_a, &mut scratch.normalized)?;
        std::mem::swap(&mut scratch.hidden_a, &mut scratch.normalized);
        Ok(())
    }
}
