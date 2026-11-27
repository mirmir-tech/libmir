//! What a projection plan compiles in beyond its geometry. Layers share
//! plans and bind their weights at execution, so a plan must refuse a weight
//! of another layout, and plans are shared only between layers whose weights
//! have the same one.

use super::{
    super::{direct_fp8::DirectFp8Layout, packed::integer::PackedIntegerLayout},
    CheckpointProjection, CheckpointProjectionWeight,
};
use crate::{
    AffineQuantizedWeight, DirectFp8CheckpointWeight, Error, MxFp4CheckpointWeight,
    PackedIntegerWeight, Result,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend) enum ProjectionLayout {
    Dense,
    /// Per output row, the packed `U32` words and the scale groups, which fix
    /// the bits and group size for the plan's input width.
    Affine {
        words: usize,
        groups: usize,
    },
    DirectFp8(DirectFp8Layout),
    MxFp4 {
        bias: bool,
    },
    MxFp8,
    /// The GGML plan dispatches on each weight's block type.
    Ggml,
    /// Prepared weights stay with their plan.
    NvFp4,
    NvFp4WeightOnly,
    PackedInteger(PackedIntegerLayout),
    /// FP8 weights quantized from BF16 at load, with their own kernels.
    BlockFp8,
    Fp8Int4,
}

impl ProjectionLayout {
    pub(in crate::backend) fn affine(weight: &AffineQuantizedWeight) -> Self {
        let trailing = |tensor: &crate::CudaTensor| tensor.shape().last().copied().unwrap_or(0);
        Self::Affine {
            words: trailing(&weight.weight),
            groups: trailing(&weight.scales),
        }
    }

    pub(in crate::backend) fn direct_fp8(weight: &DirectFp8CheckpointWeight) -> Self {
        Self::DirectFp8(weight.layout())
    }

    pub(in crate::backend) const fn mx_fp4(weight: &MxFp4CheckpointWeight) -> Self {
        Self::MxFp4 { bias: weight.has_bias() }
    }

    pub(in crate::backend) const fn packed_integer(weight: &PackedIntegerWeight) -> Self {
        Self::PackedInteger(weight.layout())
    }
}

impl CheckpointProjectionWeight {
    pub(in crate::backend) fn layout(&self) -> ProjectionLayout {
        match self {
            Self::Affine(weight) => ProjectionLayout::affine(weight),
            Self::Dense(_) => ProjectionLayout::Dense,
            Self::DirectFp8(weight) => ProjectionLayout::direct_fp8(weight),
            Self::MxFp4(weight) => ProjectionLayout::mx_fp4(weight),
            Self::Ggml(_) => ProjectionLayout::Ggml,
            Self::MxFp8(_) => ProjectionLayout::MxFp8,
            Self::NvFp4(_) => ProjectionLayout::NvFp4,
            Self::NvFp4WeightOnly(_) => ProjectionLayout::NvFp4WeightOnly,
            Self::PackedInteger(weight) => ProjectionLayout::packed_integer(weight),
        }
    }
}

impl CheckpointProjection {
    /// The layout of the weight this projection was planned for.
    fn layout(&self) -> ProjectionLayout {
        match self {
            Self::Affine { weight, .. } => ProjectionLayout::affine(weight),
            Self::Dense { .. } => ProjectionLayout::Dense,
            Self::DirectFp8 { weight, .. } => ProjectionLayout::direct_fp8(weight),
            Self::MxFp4 { weight, .. } => ProjectionLayout::mx_fp4(weight),
            Self::Ggml { .. } => ProjectionLayout::Ggml,
            Self::MxFp8 { .. } => ProjectionLayout::MxFp8,
            Self::NvFp4 { .. } => ProjectionLayout::NvFp4,
            Self::NvFp4WeightOnly { .. } => ProjectionLayout::NvFp4WeightOnly,
            Self::PackedInteger { weight, .. } => ProjectionLayout::packed_integer(weight),
        }
    }

    /// Refuses a late-bound `weight` whose layout differs from the planned
    /// one.
    pub(super) fn require_layout(&self, weight: &CheckpointProjectionWeight) -> Result<()> {
        if self.layout() == weight.layout() {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan(
                "a late-bound projection weight differs from its plan's layout",
            ))
        }
    }
}
