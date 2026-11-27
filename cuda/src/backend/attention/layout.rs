//! What an attention plan compiles in from a layer's projections beyond
//! their geometry, so layers share prefill plans only when it agrees.

use super::{DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights};
use crate::backend::linear::ProjectionLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend) struct AttentionLayout {
    qkv: QkvLayout,
    output: ProjectionLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QkvLayout {
    /// Query, key and value.
    Projected([ProjectionLayout; 3]),
    /// The query alone, of a layer reading another layer's keys and values.
    Query(ProjectionLayout),
}

impl AttentionLayout {
    pub(in crate::backend) fn of(weights: DecodeAttentionWeights<'_>) -> Self {
        type Layout = ProjectionLayout;
        let qkv = match weights.qkv {
            DecodeQkvWeights::Affine(weights) => QkvLayout::Projected(weights.map(Layout::affine)),
            DecodeQkvWeights::Bf16(_) => QkvLayout::Projected([Layout::Dense; 3]),
            DecodeQkvWeights::DirectFp8(weights) => {
                QkvLayout::Projected(weights.map(Layout::direct_fp8))
            },
            DecodeQkvWeights::MxFp4(weights) => QkvLayout::Projected(weights.map(Layout::mx_fp4)),
            DecodeQkvWeights::Ggml(_) => QkvLayout::Projected([Layout::Ggml; 3]),
            DecodeQkvWeights::MxFp8(_) => QkvLayout::Projected([Layout::MxFp8; 3]),
            DecodeQkvWeights::PackedInteger(weights) => {
                QkvLayout::Projected(weights.map(Layout::packed_integer))
            },
            DecodeQkvWeights::NvFp4(_) => QkvLayout::Projected([Layout::NvFp4; 3]),
            DecodeQkvWeights::Query(weight) => QkvLayout::Query(output(weight)),
        };
        Self { qkv, output: output(weights.output) }
    }
}

/// The layout of an attention output (or a lone query) projection.
pub(in crate::backend) fn output(weight: DecodeAttentionOutputWeight<'_>) -> ProjectionLayout {
    type Layout = ProjectionLayout;
    match weight {
        DecodeAttentionOutputWeight::Affine(weight) => Layout::affine(weight),
        DecodeAttentionOutputWeight::Bf16(_) => Layout::Dense,
        DecodeAttentionOutputWeight::DirectFp8(weight) => Layout::direct_fp8(weight),
        DecodeAttentionOutputWeight::MxFp4(weight) => Layout::mx_fp4(weight),
        DecodeAttentionOutputWeight::Ggml(_) => Layout::Ggml,
        DecodeAttentionOutputWeight::MxFp8(_) => Layout::MxFp8,
        DecodeAttentionOutputWeight::PackedInteger(weight) => Layout::packed_integer(weight),
        DecodeAttentionOutputWeight::NvFp4(_) => Layout::NvFp4,
        DecodeAttentionOutputWeight::BlockFp8 { .. } => Layout::BlockFp8,
        DecodeAttentionOutputWeight::Fp8Int4 { .. } => Layout::Fp8Int4,
    }
}
