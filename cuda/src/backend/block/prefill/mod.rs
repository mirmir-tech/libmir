mod execution;
mod scratch;

use mircuda::Stream;

use self::scratch::PrefillBlockScratch;
use super::{DecodeMoeBlockConfig, NvFp4ExpertBank, validate};
use crate::{
    Bf16Linear, Bf16LinearPair, BucketedNvFp4MoeBf16, CudaBackend, Error, ExecutionPhase,
    MoeExecution, MoePlanRequest, PrefillAttentionBf16, Result, RmsNormBf16, RouterBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16, RouterSpec},
};

/// Fixed-chunk routed-MoE prefill sharing K/V state with decode.
#[derive(Debug)]
pub struct PrefillMoeBlockBf16 {
    attention: PrefillAttentionBf16,
    post_attention_norm: RmsNormBf16,
    pre_dense_norm: RmsNormBf16,
    dense_gate_up: Bf16LinearPair,
    dense_down: Bf16Linear,
    post_dense_norm: RmsNormBf16,
    router: RouterBf16,
    pre_expert_norm: RmsNormBf16,
    experts: BucketedNvFp4MoeBf16,
    post_expert_norm: RmsNormBf16,
    post_feed_forward_norm: RmsNormBf16,
    hidden_ops: ElementwiseBf16,
    dense_activation: PackedGatedBf16,
    scratch: PrefillBlockScratch,
    stream: Stream,
    config: DecodeMoeBlockConfig,
    tokens: usize,
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_prefill_moe_block_bf16(
        &self,
        config: DecodeMoeBlockConfig,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        tokens: usize,
    ) -> Result<PrefillMoeBlockBf16> {
        PrefillMoeBlockBf16::new(self, config, gate, up, down, tokens)
    }
}

impl PrefillMoeBlockBf16 {
    fn new(
        backend: &CudaBackend,
        config: DecodeMoeBlockConfig,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
        tokens: usize,
    ) -> Result<Self> {
        validate(config)?;
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("prefill MoE block batch is empty"));
        }
        let hidden = config.attention.hidden_size;
        let dense = config.dense_intermediate;
        let epsilon = config.attention.rms_norm_epsilon;
        let hidden_elements = elements(tokens, hidden)?;
        let expert_plan = backend.execution_planner().plan_moe(MoePlanRequest::nvfp4(
            ExecutionPhase::Prefill,
            tokens,
            config.experts,
            config.top_k,
            hidden,
            config.expert_intermediate,
        ))?;
        if expert_plan.execution() != MoeExecution::Bucketed {
            return Err(Error::InvalidExecutionPlan("prefill block requires bucketed MoE"));
        }
        let norm = || RmsNormBf16::new(backend, tokens, hidden, epsilon);
        Ok(Self {
            attention: backend.prepare_prefill_attention_bf16(config.attention, tokens)?,
            post_attention_norm: norm()?,
            pre_dense_norm: norm()?,
            dense_gate_up: Bf16LinearPair::new(
                backend,
                ExecutionPhase::Prefill,
                tokens,
                hidden,
                dense,
            )?,
            dense_down: Bf16Linear::new(backend, tokens, dense, hidden)?,
            post_dense_norm: norm()?,
            router: backend.prepare_router_batch_bf16(
                RouterSpec {
                    hidden,
                    experts: config.experts,
                    top_k: config.top_k,
                    epsilon,
                    norm_multiplier: config.router_norm_multiplier,
                },
                tokens,
            )?,
            pre_expert_norm: norm()?,
            experts: backend.prepare_bucketed_nvfp4_moe_bf16(
                tokens,
                config.top_k,
                config.activation,
                gate,
                up,
                down,
            )?,
            post_expert_norm: norm()?,
            post_feed_forward_norm: norm()?,
            hidden_ops: ElementwiseBf16::compile(&backend.inner.compiler, hidden_elements)?,
            dense_activation: PackedGatedBf16::compile(&backend.inner.compiler, tokens, dense)?,
            scratch: PrefillBlockScratch::new(backend, tokens, hidden, dense)?,
            stream: backend.inner.stream.clone(),
            config,
            tokens,
        })
    }
}

fn elements(tokens: usize, width: usize) -> Result<usize> {
    tokens
        .checked_mul(width)
        .ok_or(Error::InvalidDecoderKernel("prefill MoE block size overflow"))
}
