use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    LogicalTensorRole, TensorBinding, WeightBindingPlan,
};
use crate::error::{ModelsError, Result};

#[derive(Debug)]
pub struct HybridMoeLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub attention: HybridMoeAttentionBindings<'a>,
    pub post_attention_norm: &'a TensorBinding,
    pub pre_dense_norm: &'a TensorBinding,
    pub dense: HybridMoeDenseBindings<'a>,
    pub post_dense_norm: &'a TensorBinding,
    pub router: HybridMoeRouterBindings<'a>,
    pub pre_expert_norm: &'a TensorBinding,
    pub experts: HybridMoeExpertBindings<'a>,
    pub post_expert_norm: &'a TensorBinding,
    pub post_feed_forward_norm: &'a TensorBinding,
    pub layer_scale: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridMoeAttentionBindings<'a> {
    pub query: &'a TensorBinding,
    pub key: &'a TensorBinding,
    pub value: Option<&'a TensorBinding>,
    pub output: &'a TensorBinding,
    pub query_norm: &'a TensorBinding,
    pub key_norm: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridMoeDenseBindings<'a> {
    pub gate: &'a TensorBinding,
    pub up: &'a TensorBinding,
    pub down: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridMoeRouterBindings<'a> {
    pub projection: &'a TensorBinding,
    pub norm_scale: &'a TensorBinding,
    pub expert_scale: &'a TensorBinding,
}

#[derive(Debug)]
pub enum HybridMoeExpertBindings<'a> {
    Stacked(HybridMoeDenseBindings<'a>),
    FusedStacked {
        gate_up: &'a TensorBinding,
        down: &'a TensorBinding,
    },
    Individual {
        gate: Vec<&'a TensorBinding>,
        up: Vec<&'a TensorBinding>,
        down: Vec<&'a TensorBinding>,
    },
}

impl WeightBindingPlan {
    pub fn hybrid_moe_layer(&self, index: usize) -> Result<HybridMoeLayerBindings<'_>> {
        Ok(HybridMoeLayerBindings {
            input_norm: required(self, index, LayerTensorRole::InputNorm)?,
            attention: HybridMoeAttentionBindings {
                query: attention(self, index, AttentionProjectionRole::Query)?,
                key: attention(self, index, AttentionProjectionRole::Key)?,
                value: optional_attention(self, index, AttentionProjectionRole::Value),
                output: attention(self, index, AttentionProjectionRole::Output)?,
                query_norm: required(self, index, LayerTensorRole::QueryNorm)?,
                key_norm: required(self, index, LayerTensorRole::KeyNorm)?,
            },
            post_attention_norm: required(self, index, LayerTensorRole::PostAttentionNorm)?,
            pre_dense_norm: required(self, index, LayerTensorRole::PreDenseNorm)?,
            dense: HybridMoeDenseBindings {
                gate: feed_forward(self, index, FeedForwardProjectionRole::Gate)?,
                up: feed_forward(self, index, FeedForwardProjectionRole::Up)?,
                down: feed_forward(self, index, FeedForwardProjectionRole::Down)?,
            },
            post_dense_norm: required(self, index, LayerTensorRole::PostDenseNorm)?,
            router: HybridMoeRouterBindings {
                projection: required(self, index, LayerTensorRole::Router)?,
                norm_scale: required(self, index, LayerTensorRole::RouterNormScale)?,
                expert_scale: required(self, index, LayerTensorRole::RouterExpertScale)?,
            },
            pre_expert_norm: required(self, index, LayerTensorRole::PreExpertNorm)?,
            experts: experts(self, index)?,
            post_expert_norm: required(self, index, LayerTensorRole::PostExpertNorm)?,
            post_feed_forward_norm: required(self, index, LayerTensorRole::PostFeedForwardNorm)?,
            layer_scale: required(self, index, LayerTensorRole::LayerScale)?,
        })
    }
}

impl HybridMoeLayerBindings<'_> {
    #[must_use]
    pub fn physical_sources(&self) -> Vec<&str> {
        let mut bindings = vec![
            self.input_norm,
            self.attention.query,
            self.attention.key,
            self.attention.output,
            self.attention.query_norm,
            self.attention.key_norm,
            self.post_attention_norm,
            self.pre_dense_norm,
            self.dense.gate,
            self.dense.up,
            self.dense.down,
            self.post_dense_norm,
            self.router.projection,
            self.router.norm_scale,
            self.router.expert_scale,
            self.pre_expert_norm,
            self.post_expert_norm,
            self.post_feed_forward_norm,
            self.layer_scale,
        ];
        bindings.extend(self.attention.value);
        match &self.experts {
            HybridMoeExpertBindings::Stacked(experts) => {
                bindings.extend([experts.gate, experts.up, experts.down]);
            },
            HybridMoeExpertBindings::FusedStacked { gate_up, down } => {
                bindings.extend([gate_up, down]);
            },
            HybridMoeExpertBindings::Individual { gate, up, down } => {
                bindings.extend(gate);
                bindings.extend(up);
                bindings.extend(down);
            },
        }
        bindings.into_iter().flat_map(TensorBinding::physical_sources).collect()
    }
}

fn experts(plan: &WeightBindingPlan, index: usize) -> Result<HybridMoeExpertBindings<'_>> {
    let stacked = |projection| expert(plan, index, None, projection);
    if let (Some(gate), Some(up), Some(down)) = (
        stacked(ExpertProjectionRole::Gate),
        stacked(ExpertProjectionRole::Up),
        stacked(ExpertProjectionRole::Down),
    ) {
        return Ok(HybridMoeExpertBindings::Stacked(HybridMoeDenseBindings { gate, up, down }));
    }
    if let (Some(gate_up), Some(down)) =
        (stacked(ExpertProjectionRole::GateUp), stacked(ExpertProjectionRole::Down))
    {
        return Ok(HybridMoeExpertBindings::FusedStacked { gate_up, down });
    }
    let collect = |projection| {
        let mut found =
            plan.tensors
                .iter()
                .filter_map(|binding| match binding.role {
                    LogicalTensorRole::Layer {
                        index: layer,
                        tensor:
                            LayerTensorRole::ExpertProjection {
                                expert: Some(expert),
                                projection: value,
                            },
                    } if layer == index && value == projection => Some((expert, binding)),
                    _ => None,
                })
                .collect::<Vec<_>>();
        found.sort_by_key(|(expert, _)| *expert);
        found.into_iter().map(|(_, binding)| binding).collect::<Vec<_>>()
    };
    let gate = collect(ExpertProjectionRole::Gate);
    let up = collect(ExpertProjectionRole::Up);
    let down = collect(ExpertProjectionRole::Down);
    if gate.is_empty() || gate.len() != up.len() || gate.len() != down.len() {
        return Err(invalid(index, "incomplete expert projection set"));
    }
    Ok(HybridMoeExpertBindings::Individual { gate, up, down })
}

fn attention(
    plan: &WeightBindingPlan,
    index: usize,
    projection: AttentionProjectionRole,
) -> Result<&TensorBinding> {
    required(plan, index, LayerTensorRole::AttentionProjection { projection })
}

fn optional_attention(
    plan: &WeightBindingPlan,
    index: usize,
    projection: AttentionProjectionRole,
) -> Option<&TensorBinding> {
    expert_role(plan, index, LayerTensorRole::AttentionProjection { projection })
}

fn feed_forward(
    plan: &WeightBindingPlan,
    index: usize,
    projection: FeedForwardProjectionRole,
) -> Result<&TensorBinding> {
    required(plan, index, LayerTensorRole::FeedForwardProjection { projection })
}

fn expert(
    plan: &WeightBindingPlan,
    index: usize,
    expert: Option<usize>,
    projection: ExpertProjectionRole,
) -> Option<&TensorBinding> {
    expert_role(plan, index, LayerTensorRole::ExpertProjection { expert, projection })
}

fn expert_role(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&LogicalTensorRole::Layer { index, tensor })
}

fn required(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    expert_role(plan, index, tensor)
        .ok_or_else(|| invalid(index, "required logical tensor role is unbound"))
}

fn invalid(index: usize, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid hybrid MoE layer {index}: {reason}"))
}
