use super::{
    AttentionProjectionRole, DenseFeedForwardBindings, FeedForwardProjectionRole, LayerTensorRole,
    LogicalTensorRole, RoutedExpertBindings, SharedRoutedFeedForwardBindings, TensorBinding,
    WeightBindingPlan,
};
use crate::error::{ModelsError, Result};

mod post_norm;
mod sharing;

pub use post_norm::{ParallelDecoderLayerBindings, PostNormDecoderLayerBindings};
pub use sharing::{PerLayerEmbeddingBindings, PerLayerInputBindings, SharedKeyValueLayerBindings};

#[derive(Debug, Clone, Copy)]
pub struct DenseDecoderLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub attention: DenseSoftmaxBindings<'a>,
    /// Normalizes the residual stream before the feed-forward projections.
    pub post_attention_norm: &'a TensorBinding,
    pub sandwich: Option<DenseSandwichBindings<'a>>,
    pub feed_forward: DenseLayerFeedForwardBindings<'a>,
}

/// The feed-forward of a softmax-attention decoder layer: one gated MLP,
/// experts chosen per token by a router, or those plus shared experts.
#[derive(Debug, Clone, Copy)]
pub enum DenseLayerFeedForwardBindings<'a> {
    Gated(DenseFeedForwardBindings<'a>),
    Routed(RoutedFeedForwardBindings<'a>),
    SharedRouted(SharedRoutedFeedForwardBindings<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct RoutedFeedForwardBindings<'a> {
    pub router: &'a TensorBinding,
    pub experts: RoutedExpertBindings<'a>,
}

impl<'a> DenseLayerFeedForwardBindings<'a> {
    /// The gated MLP, for loaders without routed experts.
    pub fn gated(self) -> Result<DenseFeedForwardBindings<'a>> {
        match self {
            Self::Gated(bindings) => Ok(bindings),
            Self::Routed(_) | Self::SharedRouted(_) => Err(ModelsError::InvalidConfig(
                "dense decoder layer has routed experts instead of a gated MLP".into(),
            )),
        }
    }

    /// Every binding of the feed-forward.
    #[must_use]
    pub fn bindings(self) -> Vec<&'a TensorBinding> {
        match self {
            Self::Gated(value) => vec![value.gate, value.up, value.down],
            Self::Routed(value) => {
                let mut bindings = vec![value.router];
                bindings.extend(value.experts.bindings());
                bindings
            },
            Self::SharedRouted(value) => value.bindings(),
        }
    }
}

/// Gemma-style norms on each sub-block's output before its residual add, and
/// a learned scalar on the layer output when the checkpoint has one (Gemma
/// 4; Gemma 3 has none).
#[derive(Debug, Clone, Copy)]
pub struct DenseSandwichBindings<'a> {
    pub attention_output_norm: &'a TensorBinding,
    pub feed_forward_output_norm: &'a TensorBinding,
    pub layer_scale: Option<&'a TensorBinding>,
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
        let (sandwich, post_attention_norm) = norms(self, index)?;
        Ok(DenseDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            attention: softmax(self, index)?,
            post_attention_norm,
            sandwich,
            feed_forward: feed_forward(self, index)?,
        })
    }
}

/// The Gemma sandwich norms of layer `index`, if it has them, and the norm
/// before its feed-forward.
fn norms(
    plan: &WeightBindingPlan,
    index: usize,
) -> Result<(Option<DenseSandwichBindings<'_>>, &TensorBinding)> {
    let sandwich = optional(plan, index, LayerTensorRole::PostFeedForwardNorm)
        .map(|feed_forward_output_norm| {
            Ok::<_, ModelsError>(DenseSandwichBindings {
                attention_output_norm: layer(plan, index, LayerTensorRole::PostAttentionNorm)?,
                feed_forward_output_norm,
                layer_scale: optional(plan, index, LayerTensorRole::LayerScale),
            })
        })
        .transpose()?;
    let post_attention_norm = if sandwich.is_some() {
        layer(plan, index, LayerTensorRole::PreDenseNorm)?
    } else {
        optional(plan, index, LayerTensorRole::PostAttentionNorm)
            .or_else(|| optional(plan, index, LayerTensorRole::PreDenseNorm))
            .ok_or_else(|| invalid(index, "post-attention norm is unbound"))?
    };
    Ok((sandwich, post_attention_norm))
}

impl<'a> DenseDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        self.bindings().into_iter().flat_map(TensorBinding::physical_sources).collect()
    }

    /// Every binding of the layer.
    #[must_use]
    pub fn bindings(self) -> Vec<&'a TensorBinding> {
        let mut bindings = vec![
            self.input_norm,
            self.attention.query,
            self.attention.key,
            self.attention.value,
            self.attention.output,
            self.post_attention_norm,
        ];
        bindings.extend(self.feed_forward.bindings());
        if let Some(sandwich) = self.sandwich {
            bindings.extend([sandwich.attention_output_norm, sandwich.feed_forward_output_norm]);
            bindings.extend(sandwich.layer_scale);
        }
        bindings.extend(self.attention.query_norm);
        bindings.extend(self.attention.key_norm);
        bindings
    }
}

/// The softmax attention projections and Q/K norms of layer `index`.
fn softmax(plan: &WeightBindingPlan, index: usize) -> Result<DenseSoftmaxBindings<'_>> {
    let projection = |projection| attention(plan, index, projection);
    let key = projection(AttentionProjectionRole::Key)?;
    Ok(DenseSoftmaxBindings {
        query: projection(AttentionProjectionRole::Query)?,
        key,
        // Layers whose keys double as values have no value projection.
        value: optional(
            plan,
            index,
            LayerTensorRole::AttentionProjection {
                projection: AttentionProjectionRole::Value,
            },
        )
        .unwrap_or(key),
        output: projection(AttentionProjectionRole::Output)?,
        query_norm: optional(plan, index, LayerTensorRole::QueryNorm),
        key_norm: optional(plan, index, LayerTensorRole::KeyNorm),
    })
}

/// A gated MLP, routed experts, or routed plus shared experts.
fn feed_forward(
    plan: &WeightBindingPlan,
    index: usize,
) -> Result<DenseLayerFeedForwardBindings<'_>> {
    let projection =
        |projection| layer(plan, index, LayerTensorRole::FeedForwardProjection { projection });
    let shared = LayerTensorRole::SharedExpertProjection {
        projection: FeedForwardProjectionRole::Gate,
    };
    if optional(plan, index, shared).is_some() {
        return super::hybrid::feed_forward(plan, super::slot::LayerSlot::Decoder(index))
            .map(DenseLayerFeedForwardBindings::SharedRouted);
    }
    Ok(if let Some(router) = optional(plan, index, LayerTensorRole::Router) {
        DenseLayerFeedForwardBindings::Routed(RoutedFeedForwardBindings {
            router,
            experts: super::view::experts(plan, index)?,
        })
    } else {
        DenseLayerFeedForwardBindings::Gated(DenseFeedForwardBindings {
            gate: projection(FeedForwardProjectionRole::Gate)?,
            up: projection(FeedForwardProjectionRole::Up)?,
            down: projection(FeedForwardProjectionRole::Down)?,
        })
    })
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
