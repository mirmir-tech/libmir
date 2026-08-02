mod execution;
mod layer;
mod scratch;

pub use layer::BatchedDecodeMoeLayer;
use mircuda::{DeviceBuffer, Stream, bf16};

use self::scratch::BatchBlockScratch;
use super::{
    DecodeMoeBlockConfig, DecodeMoeBlockWeights,
    experts::{ExpertWeights, Experts},
    scalar, validate,
};
use crate::{
    BatchedDecodeAttentionBf16, Bf16Linear, Bf16LinearPair, CudaBackend, Error, ExecutionPhase,
    NvFp4ExpertBank, PagedDecodeBatch, PagedKvCache, Result, RmsNormBf16, RouterBf16,
    kernels::{BatchedSplitAttentionWorkspace, ElementwiseBf16, PackedGatedBf16, RouterSpec},
};

/// Complete routed-MoE layer for independent decode rows.
#[derive(Debug)]
pub struct BatchedDecodeMoeBlockBf16 {
    attention: BatchedDecodeAttentionBf16,
    post_attention_norm: RmsNormBf16,
    pre_dense_norm: RmsNormBf16,
    dense_gate_up: Bf16LinearPair,
    dense_down: Bf16Linear,
    post_dense_norm: RmsNormBf16,
    router: RouterBf16,
    pre_expert_norm: RmsNormBf16,
    experts: Experts,
    expert_weights: ExpertWeights,
    post_expert_norm: RmsNormBf16,
    post_feed_forward_norm: RmsNormBf16,
    hidden_ops: ElementwiseBf16,
    dense_activation: PackedGatedBf16,
    scratch: BatchBlockScratch,
    stream: Stream,
    config: DecodeMoeBlockConfig,
    rows: usize,
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_batched_decode_moe_block_bf16(
        &self,
        config: DecodeMoeBlockConfig,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        rows: usize,
    ) -> Result<BatchedDecodeMoeBlockBf16> {
        BatchedDecodeMoeBlockBf16::new(self, config, gate, up, down, rows)
    }
}

impl BatchedDecodeMoeBlockBf16 {
    fn new(
        backend: &CudaBackend,
        config: DecodeMoeBlockConfig,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        rows: usize,
    ) -> Result<Self> {
        let cache = backend.prepare_paged_kv(config.attention.layer, config.attention.cache)?;
        Self::new_with_cache(
            backend,
            config,
            &ExpertWeights::NvFp4 {
                gate,
                up,
                down,
                activation_mode: models::weights::BlockActivationMode::WeightAndActivation,
            },
            rows,
            cache,
            None,
        )
    }

    pub(in crate::backend::block) fn new_with_cache(
        backend: &CudaBackend,
        config: DecodeMoeBlockConfig,
        expert_weights: &ExpertWeights,
        rows: usize,
        cache: PagedKvCache,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        validate(config)?;
        if rows == 0 {
            return Err(Error::InvalidDecoderKernel("batched decode block is empty"));
        }
        let hidden = config.attention.hidden_size;
        let dense = config.dense_intermediate;
        let epsilon = config.attention.rms_norm_epsilon;
        let norm = || RmsNormBf16::new(backend, rows, hidden, epsilon);
        let elements = rows
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("batched decode block size overflow"))?;
        Ok(Self {
            attention: BatchedDecodeAttentionBf16::new_with_cache_weights_workspace(
                backend, config.attention, rows, cache, None, workspace,
            )?,
            post_attention_norm: norm()?,
            pre_dense_norm: norm()?,
            dense_gate_up: Bf16LinearPair::new(
                backend,
                ExecutionPhase::Decode,
                rows,
                hidden,
                dense,
            )?,
            dense_down: Bf16Linear::new(backend, rows, dense, hidden)?,
            post_dense_norm: norm()?,
            router: backend.prepare_router_batch_bf16(
                RouterSpec {
                    hidden,
                    experts: config.experts,
                    top_k: config.top_k,
                    epsilon,
                    norm_multiplier: config.router_norm_multiplier,
                },
                rows,
            )?,
            pre_expert_norm: norm()?,
            experts: Experts::new(
                backend,
                ExecutionPhase::Decode,
                rows,
                config.top_k,
                config.activation,
                expert_weights,
            )?,
            expert_weights: expert_weights.clone(),
            post_expert_norm: norm()?,
            post_feed_forward_norm: norm()?,
            hidden_ops: ElementwiseBf16::compile(&backend.inner.compiler, elements)?,
            dense_activation: PackedGatedBf16::compile(&backend.inner.compiler, rows, dense)?,
            scratch: BatchBlockScratch::new(backend, rows, hidden, dense)?,
            stream: backend.inner.stream.clone(),
            config,
            rows,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if batch.active() != self.rows {
            return Err(Error::InvalidDecoderKernel("decode batch differs from block plan"));
        }
        self.attention
            .execute(input, weights.attention, batch, &mut self.scratch.attention)?;
        self.post_attention_norm.execute(
            &self.scratch.attention,
            weights.post_attention_norm,
            &mut self.scratch.attention_norm,
        )?;
        self.hidden_ops.add(
            &self.stream,
            input,
            &self.scratch.attention_norm,
            &mut self.scratch.hidden,
        )?;
        self.execute_dense(weights)?;
        self.execute_experts(weights)?;
        self.hidden_ops.add(
            &self.stream,
            &self.scratch.dense,
            &self.scratch.expert_norm,
            &mut self.scratch.feed_forward,
        )?;
        self.post_feed_forward_norm.execute(
            &self.scratch.feed_forward,
            weights.post_feed_forward_norm,
            &mut self.scratch.feed_forward_norm,
        )?;
        self.hidden_ops.add(
            &self.stream,
            &self.scratch.hidden,
            &self.scratch.feed_forward_norm,
            &mut self.scratch.residual,
        )?;
        self.hidden_ops.multiply_scalar(
            &self.stream,
            &self.scratch.residual,
            scalar(weights.layer_scalar)?,
            output,
        )
    }
}
