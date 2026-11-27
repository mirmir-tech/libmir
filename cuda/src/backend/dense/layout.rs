//! What a dense layer's prefill plan compiles in from its weights beyond
//! their geometry: layers share a plan only when every projection, expert
//! bank and per-layer input gate agrees (an MLX mixed-precision recipe stores
//! `v_proj` and `down_proj` at 6 bits in some layers and 4 in others).

use super::{DenseDownWeight, DenseFeedForwardWeights, DenseGateUpWeights, DenseSwiGluWeights};
use crate::backend::{
    attention::AttentionLayout, dense::routed::weights::RoutedLayout, linear::ProjectionLayout,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend) struct DenseLayerLayout {
    attention: AttentionLayout,
    feed_forward: FeedForwardLayout,
    /// The per-layer input gate and projection.
    per_layer: Option<[ProjectionLayout; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeedForwardLayout {
    Gated {
        gate_up: [ProjectionLayout; 2],
        down: ProjectionLayout,
    },
    Routed(RoutedLayout),
}

impl DenseLayerLayout {
    pub(in crate::backend) fn of(weights: DenseSwiGluWeights<'_>) -> Self {
        Self {
            attention: AttentionLayout::of(weights.attention),
            feed_forward: match weights.feed_forward {
                DenseFeedForwardWeights::Gated { gate_up: pair, down: weight } => {
                    FeedForwardLayout::Gated {
                        gate_up: gate_up(pair),
                        down: down(weight),
                    }
                },
                DenseFeedForwardWeights::Routed(weights) => {
                    FeedForwardLayout::Routed(weights.layout())
                },
            },
            per_layer: weights.per_layer_input.map(super::PerLayerInputWeights::layouts),
        }
    }
}

fn gate_up(weights: DenseGateUpWeights<'_>) -> [ProjectionLayout; 2] {
    type Layout = ProjectionLayout;
    match weights {
        DenseGateUpWeights::Affine { gate, up } => [Layout::affine(gate), Layout::affine(up)],
        DenseGateUpWeights::Bf16(_) => [Layout::Dense; 2],
        DenseGateUpWeights::DirectFp8 { gate, up } => {
            [Layout::direct_fp8(gate), Layout::direct_fp8(up)]
        },
        DenseGateUpWeights::Ggml { .. } => [Layout::Ggml; 2],
        DenseGateUpWeights::MxFp4 { gate, up } => [Layout::mx_fp4(gate), Layout::mx_fp4(up)],
        DenseGateUpWeights::MxFp8 { .. } => [Layout::MxFp8; 2],
        DenseGateUpWeights::PackedInteger { gate, up } => {
            [Layout::packed_integer(gate), Layout::packed_integer(up)]
        },
        DenseGateUpWeights::NvFp4 { .. } => [Layout::NvFp4; 2],
        DenseGateUpWeights::BlockFp8 { .. } => [Layout::BlockFp8; 2],
        DenseGateUpWeights::Fp8Int4 { .. } => [Layout::Fp8Int4; 2],
    }
}

fn down(weight: DenseDownWeight<'_>) -> ProjectionLayout {
    type Layout = ProjectionLayout;
    match weight {
        DenseDownWeight::Affine(weight) => Layout::affine(weight),
        DenseDownWeight::Bf16(_) => Layout::Dense,
        DenseDownWeight::DirectFp8(weight) => Layout::direct_fp8(weight),
        DenseDownWeight::Ggml(_) => Layout::Ggml,
        DenseDownWeight::MxFp4(weight) => Layout::mx_fp4(weight),
        DenseDownWeight::MxFp8(_) => Layout::MxFp8,
        DenseDownWeight::PackedInteger(weight) => Layout::packed_integer(weight),
        DenseDownWeight::NvFp4(_) => Layout::NvFp4,
        DenseDownWeight::BlockFp8 { .. } => Layout::BlockFp8,
        DenseDownWeight::Fp8Int4 { .. } => Layout::Fp8Int4,
    }
}
