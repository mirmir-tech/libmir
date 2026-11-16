use mircuda::{DeviceBuffer, bf16};

use super::super::{
    DenseFeedForwardWeights, DenseSandwichWeights, DenseSwiGluConfig, DenseSwiGluWeights,
    RoutedExpertWeights, sandwich::DenseSandwichOwned,
};
use crate::{
    AffineQuantizedWeight, Bf16LinearPackWeights, Bf16LinearPairWeights, CudaBackend, CudaTensor,
    DecodeAttentionWeights, DecodeQkvWeights, DirectFp8CheckpointWeight, GgmlCheckpointWeight,
    MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight, PackedIntegerWeight, Result,
};

mod down;
mod gate_up;
mod norm;
mod output;

use down::DenseDownOwned;
use gate_up::DenseGateUpOwned;
use norm::norm_buffer;
use output::DenseOutputOwned;

#[derive(Clone, Copy)]
pub enum DenseQkvSource<'a> {
    Affine([&'a AffineQuantizedWeight; 3]),
    Bf16(&'a Bf16LinearPackWeights<3>),
    DirectFp8([&'a DirectFp8CheckpointWeight; 3]),
    MxFp4([&'a MxFp4CheckpointWeight; 3]),
    Ggml([&'a GgmlCheckpointWeight; 3]),
    MxFp8([&'a MxFp8CheckpointWeight; 3]),
    PackedInteger([&'a PackedIntegerWeight; 3]),
    NvFp4([&'a NvFp4LinearWeight; 3]),
}

#[derive(Clone, Copy)]
pub enum DenseOutputSource<'a> {
    Affine(&'a AffineQuantizedWeight),
    Bf16(&'a CudaTensor),
    DirectFp8(&'a DirectFp8CheckpointWeight),
    MxFp4(&'a MxFp4CheckpointWeight),
    Ggml(&'a GgmlCheckpointWeight),
    MxFp8(&'a MxFp8CheckpointWeight),
    PackedInteger(&'a PackedIntegerWeight),
    NvFp4(&'a NvFp4LinearWeight),
}

#[derive(Clone, Copy)]
pub enum DenseGateUpSource<'a> {
    Affine {
        gate: &'a AffineQuantizedWeight,
        up: &'a AffineQuantizedWeight,
    },
    Bf16(&'a Bf16LinearPairWeights),
    DirectFp8 {
        gate: &'a DirectFp8CheckpointWeight,
        up: &'a DirectFp8CheckpointWeight,
    },
    MxFp4 {
        gate: &'a MxFp4CheckpointWeight,
        up: &'a MxFp4CheckpointWeight,
    },
    Ggml {
        gate: &'a GgmlCheckpointWeight,
        up: &'a GgmlCheckpointWeight,
    },
    MxFp8 {
        gate: &'a MxFp8CheckpointWeight,
        up: &'a MxFp8CheckpointWeight,
    },
    PackedInteger {
        gate: &'a PackedIntegerWeight,
        up: &'a PackedIntegerWeight,
    },
    NvFp4 {
        gate: &'a NvFp4LinearWeight,
        up: &'a NvFp4LinearWeight,
    },
}

#[derive(Clone, Copy)]
pub enum DenseDownSource<'a> {
    Affine(&'a AffineQuantizedWeight),
    Bf16(&'a CudaTensor),
    DirectFp8(&'a DirectFp8CheckpointWeight),
    MxFp4(&'a MxFp4CheckpointWeight),
    Ggml(&'a GgmlCheckpointWeight),
    MxFp8(&'a MxFp8CheckpointWeight),
    PackedInteger(&'a PackedIntegerWeight),
    NvFp4(&'a NvFp4LinearWeight),
}

#[derive(Clone, Copy)]
pub struct DenseWeightSource<'a> {
    pub input_norm: &'a CudaTensor,
    pub qkv: DenseQkvSource<'a>,
    pub query_norm: Option<&'a CudaTensor>,
    pub key_norm: Option<&'a CudaTensor>,
    pub output: DenseOutputSource<'a>,
    pub post_attention_norm: &'a CudaTensor,
    pub sandwich: Option<DenseSandwichWeights<'a>>,
    pub feed_forward: DenseFeedForwardSource<'a>,
}

#[derive(Clone, Copy)]
pub enum DenseFeedForwardSource<'a> {
    Gated {
        gate_up: DenseGateUpSource<'a>,
        down: DenseDownSource<'a>,
    },
    Routed(&'a RoutedExpertWeights),
}

#[derive(Clone)]
pub(super) struct DenseWeights {
    input_norm: CudaTensor,
    qkv: DenseQkvOwned,
    query_norm: DeviceBuffer<bf16>,
    key_norm: DeviceBuffer<bf16>,
    output: DenseOutputOwned,
    post_attention_norm: CudaTensor,
    sandwich: Option<DenseSandwichOwned>,
    feed_forward: FeedForwardOwned,
}

#[expect(
    clippy::large_enum_variant,
    reason = "one per layer executor or template; the gated MLP is the common variant"
)]
#[derive(Clone)]
enum FeedForwardOwned {
    Gated {
        gate_up: DenseGateUpOwned,
        down: DenseDownOwned,
    },
    Routed(Box<RoutedExpertWeights>),
}

#[derive(Clone)]
enum DenseQkvOwned {
    Affine(Box<[AffineQuantizedWeight; 3]>),
    Bf16(Bf16LinearPackWeights<3>),
    DirectFp8(Box<[DirectFp8CheckpointWeight; 3]>),
    MxFp4(Box<[MxFp4CheckpointWeight; 3]>),
    Ggml(Box<[GgmlCheckpointWeight; 3]>),
    MxFp8(Box<[MxFp8CheckpointWeight; 3]>),
    PackedInteger(Box<[PackedIntegerWeight; 3]>),
    NvFp4([NvFp4LinearWeight; 3]),
}

impl DenseWeights {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        source: DenseWeightSource<'_>,
    ) -> Result<Self> {
        let head_dim = config.attention.cache.key_head_dim;
        Ok(Self {
            input_norm: source.input_norm.clone(),
            qkv: source.qkv.into(),
            query_norm: norm_buffer(
                backend,
                source.query_norm,
                config.attention.qkv_normalization.query,
                head_dim,
            )?,
            key_norm: norm_buffer(
                backend,
                source.key_norm,
                config.attention.qkv_normalization.key,
                head_dim,
            )?,
            output: DenseOutputOwned::new(backend, source.output)?,
            post_attention_norm: source.post_attention_norm.clone(),
            sandwich: source.sandwich.map(DenseSandwichOwned::new).transpose()?,
            feed_forward: match source.feed_forward {
                DenseFeedForwardSource::Gated { gate_up, down } => FeedForwardOwned::Gated {
                    gate_up: DenseGateUpOwned::new(backend, gate_up)?,
                    down: DenseDownOwned::new(backend, down)?,
                },
                DenseFeedForwardSource::Routed(weights) => {
                    FeedForwardOwned::Routed(Box::new(weights.clone()))
                },
            },
        })
    }

    pub(super) fn borrow(&self) -> DenseSwiGluWeights<'_> {
        DenseSwiGluWeights {
            attention: DecodeAttentionWeights {
                input_norm: &self.input_norm,
                qkv: self.qkv.borrow(),
                query_norm: &self.query_norm,
                key_norm: &self.key_norm,
                output: self.output.borrow(),
            },
            post_attention_norm: &self.post_attention_norm,
            sandwich: self.sandwich.as_ref().map(DenseSandwichOwned::borrow),
            feed_forward: match &self.feed_forward {
                FeedForwardOwned::Gated { gate_up, down } => DenseFeedForwardWeights::Gated {
                    gate_up: gate_up.borrow(),
                    down: down.borrow(),
                },
                FeedForwardOwned::Routed(weights) => DenseFeedForwardWeights::Routed(weights),
            },
        }
    }
}

impl DenseQkvOwned {
    fn borrow(&self) -> DecodeQkvWeights<'_> {
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
        }
    }
}
