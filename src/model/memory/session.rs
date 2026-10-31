//! Per-sequence recurrent state that is not part of the paged K/V cache.

use models::layout::{AttentionLayerType, DecoderConfig, StateSpaceStateDType};

/// Bytes of recurrent state one sequence keeps resident: gated-delta state
/// plus convolution history, and Mamba-2 SSM state plus convolution history.
pub(super) fn state_bytes(decoder: &DecoderConfig) -> u64 {
    gated_delta_bytes(decoder).saturating_add(state_space_bytes(decoder))
}

/// Float32 recurrent state plus BF16 convolution history of every linear
/// layer.
fn gated_delta_bytes(decoder: &DecoderConfig) -> u64 {
    let Some(linear) = decoder.linear_attention.as_ref() else {
        return 0;
    };
    let layers = layer_count(decoder, AttentionLayerType::Linear);
    let state = linear.value_heads * linear.value_head_dim * linear.key_head_dim * 4;
    let channels =
        2 * linear.key_heads * linear.key_head_dim + linear.value_heads * linear.value_head_dim;
    let history = channels * linear.convolution_kernel_size.saturating_sub(1) * 2;
    u64::try_from(layers * (state + history)).unwrap_or(u64::MAX)
}

/// SSM state in its configured element type plus BF16 convolution history of
/// every state-space layer.
fn state_space_bytes(decoder: &DecoderConfig) -> u64 {
    let Some(hybrid) = decoder.state_space.as_ref() else {
        return 0;
    };
    let mamba = &hybrid.mamba;
    let layers = layer_count(decoder, AttentionLayerType::StateSpace);
    let element = match mamba.state_dtype {
        StateSpaceStateDType::Float32 => 4,
        StateSpaceStateDType::BFloat16 | StateSpaceStateDType::Float16 => 2,
    };
    let state = mamba.heads * mamba.head_dim * mamba.state_size * element;
    let history =
        mamba.convolution_channels() * mamba.convolution_kernel_size.saturating_sub(1) * 2;
    u64::try_from(layers.saturating_mul(state.saturating_add(history))).unwrap_or(u64::MAX)
}

fn layer_count(decoder: &DecoderConfig, kind: AttentionLayerType) -> usize {
    decoder.layer_types.iter().filter(|layer| **layer == kind).count()
}
