use super::DecodeMoeBlockWeights;
use crate::{
    Bf16LinearPairWeights, CudaTensor, RouterTensors,
    backend::attention::graph::CapturedAttentionWeights,
};

#[derive(Clone)]
pub(in crate::backend) struct CapturedBlockWeights {
    attention: CapturedAttentionWeights,
    post_attention_norm: CudaTensor,
    pre_dense_norm: CudaTensor,
    dense_gate_up: Bf16LinearPairWeights,
    dense_down: CudaTensor,
    post_dense_norm: CudaTensor,
    router_projection: CudaTensor,
    router_norm_scale: CudaTensor,
    router_expert_scale: CudaTensor,
    pre_expert_norm: CudaTensor,
    post_expert_norm: CudaTensor,
    post_feed_forward_norm: CudaTensor,
    layer_scalar: CudaTensor,
}

impl CapturedBlockWeights {
    pub(in crate::backend) fn prepare(
        backend: &crate::CudaBackend,
        weights: DecodeMoeBlockWeights<'_>,
    ) -> crate::Result<Self> {
        Ok(Self {
            attention: CapturedAttentionWeights::prepare(backend, weights.attention)?,
            post_attention_norm: weights.post_attention_norm.clone(),
            pre_dense_norm: weights.pre_dense_norm.clone(),
            dense_gate_up: weights.dense_gate_up.clone(),
            dense_down: weights.dense_down.clone(),
            post_dense_norm: weights.post_dense_norm.clone(),
            router_projection: weights.router.projection.clone(),
            router_norm_scale: weights.router.norm_scale.clone(),
            router_expert_scale: weights.router.expert_scale.clone(),
            pre_expert_norm: weights.pre_expert_norm.clone(),
            post_expert_norm: weights.post_expert_norm.clone(),
            post_feed_forward_norm: weights.post_feed_forward_norm.clone(),
            layer_scalar: weights.layer_scalar.clone(),
        })
    }

    pub(in crate::backend::block) fn borrow(&self) -> DecodeMoeBlockWeights<'_> {
        DecodeMoeBlockWeights {
            attention: self.attention.borrow(),
            post_attention_norm: &self.post_attention_norm,
            pre_dense_norm: &self.pre_dense_norm,
            dense_gate_up: &self.dense_gate_up,
            dense_down: &self.dense_down,
            post_dense_norm: &self.post_dense_norm,
            router: RouterTensors {
                projection: &self.router_projection,
                norm_scale: &self.router_norm_scale,
                expert_scale: &self.router_expert_scale,
            },
            pre_expert_norm: &self.pre_expert_norm,
            post_expert_norm: &self.post_expert_norm,
            post_feed_forward_norm: &self.post_feed_forward_norm,
            layer_scalar: &self.layer_scalar,
        }
    }
}

impl From<DecodeMoeBlockWeights<'_>> for CapturedBlockWeights {
    fn from(weights: DecodeMoeBlockWeights<'_>) -> Self {
        Self {
            attention: weights.attention.into(),
            post_attention_norm: weights.post_attention_norm.clone(),
            pre_dense_norm: weights.pre_dense_norm.clone(),
            dense_gate_up: weights.dense_gate_up.clone(),
            dense_down: weights.dense_down.clone(),
            post_dense_norm: weights.post_dense_norm.clone(),
            router_projection: weights.router.projection.clone(),
            router_norm_scale: weights.router.norm_scale.clone(),
            router_expert_scale: weights.router.expert_scale.clone(),
            pre_expert_norm: weights.pre_expert_norm.clone(),
            post_expert_norm: weights.post_expert_norm.clone(),
            post_feed_forward_norm: weights.post_feed_forward_norm.clone(),
            layer_scalar: weights.layer_scalar.clone(),
        }
    }
}
