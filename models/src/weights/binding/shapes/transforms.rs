//! Layout transforms a binding applies between checkpoint and logical shapes.

use super::{clamped, routed};
use crate::{
    semantic::{DecoderLayerSpec, MixerSpec, SemanticModelSpec},
    weights::binding::{
        AttentionProjectionRole, BindingTransform, ExpertProjectionRole, LayerTensorRole,
        LogicalTensorRole, TensorPacking, TensorStorage, dimensions,
    },
};

pub(in crate::weights::binding) fn transforms(
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
        ) && let Some(DecoderLayerSpec {
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
