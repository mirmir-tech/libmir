//! Logical shapes of multi-head latent attention tensors.

use super::{AttentionProjectionRole, LatentAttentionTensorRole};
use crate::semantic::MixerSpec;

/// The direct query (`q_proj`, checkpoints without a query latent) and the
/// output projection of a latent-attention layer.
pub(super) fn projection(
    mixer: &MixerSpec,
    hidden: usize,
    projection: AttentionProjectionRole,
) -> Option<Vec<usize>> {
    let MixerSpec::LatentAttention(latent) = mixer else {
        return None;
    };
    let query = latent.query_heads.checked_mul(latent.query_head_dim())?;
    match projection {
        AttentionProjectionRole::Query if latent.query_rank.is_none() => Some(vec![query, hidden]),
        AttentionProjectionRole::Output => {
            Some(vec![hidden, latent.query_heads.checked_mul(latent.value_head_dim)?])
        },
        AttentionProjectionRole::Query
        | AttentionProjectionRole::Key
        | AttentionProjectionRole::Value
        | AttentionProjectionRole::Qkv => None,
    }
}

pub(super) fn tensor(
    mixer: &MixerSpec,
    hidden: usize,
    tensor: LatentAttentionTensorRole,
) -> Option<Vec<usize>> {
    let MixerSpec::LatentAttention(latent) = mixer else {
        return None;
    };
    let query = latent.query_heads.checked_mul(latent.query_head_dim())?;
    let key_value = latent
        .query_heads
        .checked_mul(latent.nope_head_dim.checked_add(latent.value_head_dim)?)?;
    Some(match tensor {
        LatentAttentionTensorRole::QueryDown => vec![latent.query_rank?, hidden],
        LatentAttentionTensorRole::QueryLatentNorm => vec![latent.query_rank?],
        LatentAttentionTensorRole::QueryUp => vec![query, latent.query_rank?],
        LatentAttentionTensorRole::KeyValueDown => vec![latent.cached_width(), hidden],
        LatentAttentionTensorRole::KeyValueLatentNorm => vec![latent.key_value_rank],
        LatentAttentionTensorRole::KeyValueUp => vec![key_value, latent.key_value_rank],
    })
}
