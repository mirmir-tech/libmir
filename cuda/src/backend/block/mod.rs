mod batch;
mod config;
mod execution;
pub(in crate::backend) mod experts;
mod graph;
mod prefill;
mod runner;
mod scratch;
mod template;
#[cfg(test)]
mod tests;
mod trace;

use ::runtime::kv::{BlockTable, KvWritePlan};
pub use batch::{BatchedDecodeMoeBlockBf16, BatchedDecodeMoeLayer};
use config::validate;
pub use config::{DecodeMoeBlockConfig, DecodeMoeBlockWeights};
pub use graph::CapturedDecodeMoeBlockBf16;
pub(in crate::backend) use graph::CapturedLayer;
use mircuda::{DeviceBuffer, Stream, bf16};
pub use prefill::PrefillMoeBlockBf16;
pub(in crate::backend) use runner::PreparedDecodeMoeBlock;
pub use runner::{DecodeGraphAction, DecodeMoeBlockExecutor};
pub use template::DecodeMoeLayerTemplate;

use self::{
    experts::{ExpertWeights, Experts},
    scratch::BlockScratch,
};
use super::{
    Bf16Linear, Bf16LinearPair, Bf16LinearPairWeights, CudaBackend, DecodeAttentionBf16,
    DecodeAttentionConfig, DecodeAttentionWeights, ExecutionPhase, GatedActivation,
    NvFp4ExpertBank, PagedKvCache, RmsNormBf16, RouterBf16, RouterTensors,
};
use crate::{
    CudaTensor, Error, Result,
    kernels::{ElementwiseBf16, PackedGatedBf16, RouterSpec},
};

#[derive(Debug)]
pub struct DecodeMoeBlockBf16 {
    attention: DecodeAttentionBf16,
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
    scratch: BlockScratch,
    stream: Stream,
    config: DecodeMoeBlockConfig,
}

impl CudaBackend {
    pub fn prepare_decode_moe_block_bf16(
        &self,
        config: DecodeMoeBlockConfig,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<DecodeMoeBlockBf16> {
        DecodeMoeBlockBf16::new(self, config, gate, up, down)
    }
}

impl DecodeMoeBlockBf16 {
    fn new(
        backend: &CudaBackend,
        config: DecodeMoeBlockConfig,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
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
            cache,
        )
    }

    pub(in crate::backend::block) fn new_with_cache(
        backend: &CudaBackend,
        config: DecodeMoeBlockConfig,
        expert_weights: &ExpertWeights,
        cache: PagedKvCache,
    ) -> Result<Self> {
        validate(config)?;
        let hidden = config.attention.hidden_size;
        let epsilon = config.attention.rms_norm_epsilon;
        trace::prepared(config);
        let norm = || RmsNormBf16::new(backend, 1, hidden, epsilon);
        Ok(Self {
            attention: DecodeAttentionBf16::new_with_cache(backend, config.attention, cache)?,
            post_attention_norm: norm()?,
            pre_dense_norm: norm()?,
            dense_gate_up: Bf16LinearPair::new(
                backend,
                ExecutionPhase::Decode,
                1,
                hidden,
                config.dense_intermediate,
            )?,
            dense_down: Bf16Linear::new(backend, 1, config.dense_intermediate, hidden)?,
            post_dense_norm: norm()?,
            router: backend.prepare_router_bf16(RouterSpec {
                hidden,
                experts: config.experts,
                top_k: config.top_k,
                epsilon,
                norm_multiplier: config.router_norm_multiplier,
            })?,
            pre_expert_norm: norm()?,
            experts: Experts::new(
                backend,
                ExecutionPhase::Decode,
                1,
                config.top_k,
                config.activation,
                expert_weights,
            )?,
            expert_weights: expert_weights.clone(),
            post_expert_norm: norm()?,
            post_feed_forward_norm: norm()?,
            hidden_ops: ElementwiseBf16::compile(&backend.inner.compiler, hidden)?,
            dense_activation: PackedGatedBf16::compile(
                &backend.inner.compiler,
                1,
                config.dense_intermediate,
            )?,
            scratch: BlockScratch::new(backend, hidden, config.dense_intermediate)?,
            stream: backend.inner.stream.clone(),
            config,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute(
            input,
            weights.attention,
            write_plan,
            table,
            &mut self.scratch.attention,
        )?;
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

    /// Transfers this layer into a session-local direct/graph executor.
    #[must_use]
    pub fn into_executor(
        self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        output: &DeviceBuffer<bf16>,
    ) -> DecodeMoeBlockExecutor {
        DecodeMoeBlockExecutor::new(self, input.clone(), weights.into(), output.clone())
    }
}

fn scalar(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    if tensor.shape() != [1] {
        return Err(Error::InvalidDecoderKernel("layer scalar must contain one BF16 value"));
    }
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
