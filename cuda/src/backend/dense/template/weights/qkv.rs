//! A layer's owned query, key and value projections.

use super::{DenseQkvSource, output::DenseOutputOwned};
use crate::{
    AffineQuantizedWeight, Bf16LinearPackWeights, DecodeQkvWeights, DirectFp8CheckpointWeight,
    GgmlCheckpointWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight,
    PackedIntegerWeight,
};

#[derive(Clone)]
pub(super) enum DenseQkvOwned {
    Affine(Box<[AffineQuantizedWeight; 3]>),
    Bf16(Bf16LinearPackWeights<3>),
    DirectFp8(Box<[DirectFp8CheckpointWeight; 3]>),
    MxFp4(Box<[MxFp4CheckpointWeight; 3]>),
    Ggml(Box<[GgmlCheckpointWeight; 3]>),
    MxFp8(Box<[MxFp8CheckpointWeight; 3]>),
    PackedInteger(Box<[PackedIntegerWeight; 3]>),
    NvFp4([NvFp4LinearWeight; 3]),
    /// The query projection of a layer reading another's keys and values.
    Query(Box<DenseOutputOwned>),
}

impl DenseQkvOwned {
    pub(super) fn borrow(&self) -> DecodeQkvWeights<'_> {
        match self {
            Self::Affine(weights) => {
                DecodeQkvWeights::Affine([&weights[0], &weights[1], &weights[2]])
            },
            Self::Bf16(weights) => DecodeQkvWeights::Bf16(weights),
            Self::DirectFp8(weights) => {
                DecodeQkvWeights::DirectFp8([&weights[0], &weights[1], &weights[2]])
            },
            Self::MxFp4(weights) => {
                DecodeQkvWeights::MxFp4([&weights[0], &weights[1], &weights[2]])
            },
            Self::Ggml(weights) => DecodeQkvWeights::Ggml([&weights[0], &weights[1], &weights[2]]),
            Self::MxFp8(weights) => {
                DecodeQkvWeights::MxFp8([&weights[0], &weights[1], &weights[2]])
            },
            Self::PackedInteger(weights) => {
                DecodeQkvWeights::PackedInteger([&weights[0], &weights[1], &weights[2]])
            },
            Self::NvFp4(weights) => {
                DecodeQkvWeights::NvFp4([&weights[0], &weights[1], &weights[2]])
            },
            Self::Query(weight) => DecodeQkvWeights::Query(weight.borrow()),
        }
    }
}

impl From<DenseQkvSource<'_>> for DenseQkvOwned {
    fn from(source: DenseQkvSource<'_>) -> Self {
        match source {
            DenseQkvSource::Affine(weights) => Self::Affine(Box::new(weights.map(Clone::clone))),
            DenseQkvSource::Bf16(weights) => Self::Bf16(weights.clone()),
            DenseQkvSource::DirectFp8(weights) => {
                Self::DirectFp8(Box::new(weights.map(Clone::clone)))
            },
            DenseQkvSource::MxFp4(weights) => Self::MxFp4(Box::new(weights.map(Clone::clone))),
            DenseQkvSource::Ggml(weights) => Self::Ggml(Box::new(weights.map(Clone::clone))),
            DenseQkvSource::MxFp8(weights) => Self::MxFp8(Box::new(weights.map(Clone::clone))),
            DenseQkvSource::PackedInteger(weights) => {
                Self::PackedInteger(Box::new(weights.map(Clone::clone)))
            },
            DenseQkvSource::NvFp4(weights) => Self::NvFp4(weights.map(Clone::clone)),
            DenseQkvSource::Query(weight) => Self::Query(Box::new(DenseOutputOwned::query(weight))),
        }
    }
}
