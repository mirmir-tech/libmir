use super::{
    AttentionProjectionRole, FeedForwardProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, WeightBindingPlan,
};
use crate::error::{ModelsError, Result};

#[derive(Debug, Clone, Copy)]
pub struct DenseDecoderLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub attention: DenseSoftmaxBindings<'a>,
    pub post_attention_norm: &'a TensorBinding,
    pub gate: &'a TensorBinding,
    pub up: &'a TensorBinding,
    pub down: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct DenseSoftmaxBindings<'a> {
    pub query: &'a TensorBinding,
    pub key: &'a TensorBinding,
    pub value: &'a TensorBinding,
    pub output: &'a TensorBinding,
    pub query_norm: Option<&'a TensorBinding>,
    pub key_norm: Option<&'a TensorBinding>,
}

impl WeightBindingPlan {
    pub fn dense_decoder_layer(&self, index: usize) -> Result<DenseDecoderLayerBindings<'_>> {
        let projection = |projection| attention(self, index, projection);
        let feed_forward =
            |projection| layer(self, index, LayerTensorRole::FeedForwardProjection { projection });
        Ok(DenseDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            attention: DenseSoftmaxBindings {
                query: projection(AttentionProjectionRole::Query)?,
                key: projection(AttentionProjectionRole::Key)?,
                value: projection(AttentionProjectionRole::Value)?,
                output: projection(AttentionProjectionRole::Output)?,
                query_norm: optional(self, index, LayerTensorRole::QueryNorm),
                key_norm: optional(self, index, LayerTensorRole::KeyNorm),
            },
            post_attention_norm: optional(self, index, LayerTensorRole::PostAttentionNorm)
                .or_else(|| optional(self, index, LayerTensorRole::PreDenseNorm))
                .ok_or_else(|| invalid(index, "post-attention norm is unbound"))?,
            gate: feed_forward(FeedForwardProjectionRole::Gate)?,
            up: feed_forward(FeedForwardProjectionRole::Up)?,
            down: feed_forward(FeedForwardProjectionRole::Down)?,
        })
    }
}

impl<'a> DenseDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = vec![
            self.input_norm,
            self.attention.query,
            self.attention.key,
            self.attention.value,
            self.attention.output,
            self.post_attention_norm,
            self.gate,
            self.up,
            self.down,
        ];
        bindings.extend(self.attention.query_norm);
        bindings.extend(self.attention.key_norm);
        bindings.into_iter().flat_map(TensorBinding::physical_sources).collect()
    }
}

fn attention(
    plan: &WeightBindingPlan,
    index: usize,
    projection: AttentionProjectionRole,
) -> Result<&TensorBinding> {
    layer(plan, index, LayerTensorRole::AttentionProjection { projection })
}

fn layer(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    let role = LogicalTensorRole::Layer { index, tensor };
    plan.binding(&role).ok_or_else(|| invalid(index, &format!("unbound {role:?}")))
}

fn optional(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Option<&TensorBinding> {
    plan.binding(&LogicalTensorRole::Layer { index, tensor })
}

fn invalid(index: usize, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid dense decoder layer {index}: {reason}"))
}
