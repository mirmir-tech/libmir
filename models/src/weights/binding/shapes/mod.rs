use super::{
    AttentionProjectionRole, BindingTransform, ExpertProjectionRole, FeedForwardProjectionRole,
    LayerTensorRole, LinearAttentionTensorRole, LogicalTensorRole, TensorPacking, TensorStorage,
    dimensions,
};
use crate::semantic::{AttentionOutputSpec, FeedForwardSpec, MixerSpec, SemanticModelSpec};

mod clamped;
mod feed_forward;

use feed_forward::{routed, shared};

pub(super) fn logical(
    spec: &SemanticModelSpec,
    role: &LogicalTensorRole,
    vision_projection: Option<[usize; 2]>,
) -> Option<Vec<usize>> {
    match role {
        LogicalTensorRole::Embedding => {
            Some(vec![spec.decoder.vocab_size, spec.decoder.hidden_size])
        },
        LogicalTensorRole::FinalNorm => Some(vec![spec.decoder.hidden_size]),
        LogicalTensorRole::Output => Some(vec![spec.decoder.vocab_size, spec.decoder.hidden_size]),
        LogicalTensorRole::VisionProjection => vision_projection.map(Vec::from),
        LogicalTensorRole::Layer { index, tensor } => layer(spec, *index, tensor),
        LogicalTensorRole::Auxiliary { .. } => None,
    }
}

fn layer(spec: &SemanticModelSpec, index: usize, role: &LayerTensorRole) -> Option<Vec<usize>> {
    let layer = spec.decoder.layers.get(index)?;
    let hidden = spec.decoder.hidden_size;
    match role {
        LayerTensorRole::InputNorm
        | LayerTensorRole::PostAttentionNorm
        | LayerTensorRole::PreDenseNorm
        | LayerTensorRole::PostDenseNorm
        | LayerTensorRole::PreExpertNorm
        | LayerTensorRole::PostExpertNorm
        | LayerTensorRole::PostFeedForwardNorm
        | LayerTensorRole::RouterNormScale => Some(vec![hidden]),
        LayerTensorRole::RouterExpertScale => {
            routed(&layer.feed_forward).map(|routed| vec![routed.expert_count])
        },
        LayerTensorRole::LayerScale => Some(vec![1]),
        LayerTensorRole::AttentionProjection { projection } => {
            let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
                return None;
            };
            let query = attention.query_heads.checked_mul(attention.head_dim)?;
            let key_value = attention.key_value_heads.checked_mul(attention.head_dim)?;
            Some(match projection {
                AttentionProjectionRole::Query => vec![
                    if attention.output == AttentionOutputSpec::Gated {
                        query.checked_mul(2)?
                    } else {
                        query
                    },
                    hidden,
                ],
                AttentionProjectionRole::Key | AttentionProjectionRole::Value => {
                    vec![key_value, hidden]
                },
                AttentionProjectionRole::Qkv => {
                    vec![query.checked_add(key_value.checked_mul(2)?)?, hidden]
                },
                AttentionProjectionRole::Output => vec![hidden, query],
            })
        },
        LayerTensorRole::QueryNorm | LayerTensorRole::KeyNorm => {
            let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
                return None;
            };
            Some(vec![attention.head_dim])
        },
        LayerTensorRole::AttentionSinks => {
            let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
                return None;
            };
            Some(vec![attention.query_heads])
        },
        LayerTensorRole::LinearAttention { tensor } => {
            linear_attention(&layer.mixer, hidden, *tensor)
        },
        LayerTensorRole::Router => router(&layer.feed_forward, hidden),
        LayerTensorRole::FeedForwardProjection { projection } => {
            dense(&layer.feed_forward, hidden, *projection)
        },
        LayerTensorRole::ExpertProjection { expert, projection } => {
            experts(&layer.feed_forward, hidden, *expert, *projection)
        },
        LayerTensorRole::SharedExpertProjection { projection } => {
            shared_expert(&layer.feed_forward, hidden, *projection)
        },
        LayerTensorRole::SharedExpertOutputGate => shared_output_gate(&layer.feed_forward, hidden),
        LayerTensorRole::Auxiliary { .. } => None,
    }
}

fn linear_attention(
    mixer: &MixerSpec,
    hidden: usize,
    tensor: LinearAttentionTensorRole,
) -> Option<Vec<usize>> {
    let MixerSpec::LinearAttention(linear) = mixer else {
        return None;
    };
    let key = linear.key_heads.checked_mul(linear.key_head_dim)?;
    let value = linear.value_heads.checked_mul(linear.value_head_dim)?;
    let mixed_width = key.checked_mul(2)?.checked_add(value)?;
    Some(match tensor {
        LinearAttentionTensorRole::DecayLog | LinearAttentionTensorRole::TimeBias => {
            vec![linear.value_heads]
        },
        LinearAttentionTensorRole::Convolution => {
            vec![mixed_width, linear.convolution_kernel_size, 1]
        },
        LinearAttentionTensorRole::QkvProjection => vec![mixed_width, hidden],
        LinearAttentionTensorRole::GateProjection => vec![value, hidden],
        LinearAttentionTensorRole::AlphaProjection | LinearAttentionTensorRole::BetaProjection => {
            vec![linear.value_heads, hidden]
        },
        LinearAttentionTensorRole::Norm => vec![linear.value_head_dim],
        LinearAttentionTensorRole::OutputProjection => vec![hidden, value],
    })
}

pub(super) fn transforms(
    spec: &SemanticModelSpec,
    role: &LogicalTensorRole,
    source: &str,
    physical: &[usize],
    logical: Option<&[usize]>,
    storage: &TensorStorage,
) -> Vec<BindingTransform> {
    let mut transforms = Vec::new();
    if let LogicalTensorRole::Layer { index, tensor } = role {
        let layer = spec.decoder.layers.get(*index);
        if matches!(
            tensor,
            LayerTensorRole::AttentionProjection { projection: AttentionProjectionRole::Qkv }
        ) && let Some(crate::semantic::DecoderLayerSpec {
            mixer: MixerSpec::SoftmaxAttention(attention),
            ..
        }) = layer
        {
            transforms.push(BindingTransform::FusedQkv {
                query: attention.query_heads * attention.head_dim,
                key: attention.key_value_heads * attention.head_dim,
                value: attention.key_value_heads * attention.head_dim,
            });
        }
        if let LayerTensorRole::ExpertProjection { expert: None, projection } = tensor {
            if let Some(routed) = layer.and_then(|layer| routed(&layer.feed_forward)) {
                transforms.push(BindingTransform::StackedExperts { count: routed.expert_count });
            }
            if *projection == ExpertProjectionRole::GateUp {
                let interleaved = matches!(
                    storage,
                    TensorStorage::BlockQuantized {
                        packing: TensorPacking::InterleavedGateUp,
                        ..
                    }
                ) || clamped::dense_expert(spec, role, source);
                transforms.push(BindingTransform::FusedGateUp { interleaved });
            }
        }
    }
    if matches!(storage, TensorStorage::Dense { .. } | TensorStorage::Float8 { .. })
        && (clamped::dense_expert(spec, role, source)
            || logical.is_some_and(|shape| dimensions::transposes_last_two(shape, physical)))
    {
        transforms.push(BindingTransform::Transpose);
    }
    transforms
}

fn router(feed_forward: &FeedForwardSpec, hidden: usize) -> Option<Vec<usize>> {
    let routed = routed(feed_forward)?;
    Some(vec![routed.expert_count, hidden])
}

fn dense(
    feed_forward: &FeedForwardSpec,
    hidden: usize,
    projection: FeedForwardProjectionRole,
) -> Option<Vec<usize>> {
    let intermediate_size = match feed_forward {
        FeedForwardSpec::Dense { intermediate_size, .. } => *intermediate_size,
        FeedForwardSpec::DenseAndRouted { dense_intermediate_size, .. } => *dense_intermediate_size,
        FeedForwardSpec::Routed { .. } => return None,
    };
    Some(match projection {
        FeedForwardProjectionRole::Gate | FeedForwardProjectionRole::Up => {
            vec![intermediate_size, hidden]
        },
        FeedForwardProjectionRole::Down => vec![hidden, intermediate_size],
    })
}

fn experts(
    feed_forward: &FeedForwardSpec,
    hidden: usize,
    expert: Option<usize>,
    projection: ExpertProjectionRole,
) -> Option<Vec<usize>> {
    let routed = routed(feed_forward)?;
    let output = match projection {
        ExpertProjectionRole::Gate | ExpertProjectionRole::Up => routed.intermediate_size,
        ExpertProjectionRole::GateUp => routed.intermediate_size.checked_mul(2)?,
        ExpertProjectionRole::Down => hidden,
    };
    let input = if projection == ExpertProjectionRole::Down {
        routed.intermediate_size
    } else {
        hidden
    };
    match expert {
        Some(index) if index < routed.expert_count => Some(vec![output, input]),
        Some(_) => None,
        None => Some(vec![routed.expert_count, output, input]),
    }
}

fn shared_expert(
    feed_forward: &FeedForwardSpec,
    hidden: usize,
    projection: FeedForwardProjectionRole,
) -> Option<Vec<usize>> {
    let shared = shared(feed_forward)?;
    Some(match projection {
        FeedForwardProjectionRole::Gate | FeedForwardProjectionRole::Up => {
            vec![shared.intermediate_size, hidden]
        },
        FeedForwardProjectionRole::Down => vec![hidden, shared.intermediate_size],
    })
}

fn shared_output_gate(feed_forward: &FeedForwardSpec, hidden: usize) -> Option<Vec<usize>> {
    shared(feed_forward)?.gated_output.then_some(vec![1, hidden])
}
