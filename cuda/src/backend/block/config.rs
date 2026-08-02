use super::{
    Bf16LinearPairWeights, CudaTensor, DecodeAttentionConfig, DecodeAttentionWeights,
    GatedActivation, RouterTensors,
};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeMoeBlockConfig {
    pub attention: DecodeAttentionConfig,
    pub dense_intermediate: usize,
    pub expert_intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub router_norm_multiplier: f32,
    pub activation: GatedActivation,
}

#[derive(Clone, Copy)]
pub struct DecodeMoeBlockWeights<'a> {
    pub attention: DecodeAttentionWeights<'a>,
    pub post_attention_norm: &'a CudaTensor,
    pub pre_dense_norm: &'a CudaTensor,
    pub dense_gate_up: &'a Bf16LinearPairWeights,
    pub dense_down: &'a CudaTensor,
    pub post_dense_norm: &'a CudaTensor,
    pub router: RouterTensors<'a>,
    pub pre_expert_norm: &'a CudaTensor,
    pub post_expert_norm: &'a CudaTensor,
    pub post_feed_forward_norm: &'a CudaTensor,
    pub layer_scalar: &'a CudaTensor,
}

pub(super) fn validate(config: DecodeMoeBlockConfig) -> Result<()> {
    if config.dense_intermediate == 0
        || config.expert_intermediate == 0
        || config.experts == 0
        || config.top_k == 0
        || config.top_k > config.experts
        || !config.router_norm_multiplier.is_finite()
    {
        Err(Error::InvalidDecoderKernel("invalid routed MoE block configuration"))
    } else {
        Ok(())
    }
}
