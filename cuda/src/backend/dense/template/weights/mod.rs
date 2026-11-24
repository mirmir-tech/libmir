use mircuda::{DeviceBuffer, bf16};

use super::super::{
    DenseFeedForwardWeights, DenseSandwichWeights, DenseSwiGluConfig, DenseSwiGluWeights,
    PerLayerInputWeights, RoutedExpertWeights, per_layer::PerLayerInputOwned,
    sandwich::DenseSandwichOwned,
};
use crate::{
    AffineQuantizedWeight, Bf16LinearPackWeights, Bf16LinearPairWeights, CudaBackend, CudaTensor,
    DecodeAttentionWeights, DirectFp8CheckpointWeight, GgmlCheckpointWeight, MxFp4CheckpointWeight,
    MxFp8CheckpointWeight, NvFp4LinearWeight, PackedIntegerWeight, Result,
};

mod down;
mod gate_up;
mod norm;
mod output;
mod qkv;

use down::DenseDownOwned;
use gate_up::DenseGateUpOwned;
use norm::qk_norms;
use output::DenseOutputOwned;
use qkv::DenseQkvOwned;

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
    /// The query projection alone, of a layer that attends over another
    /// layer's keys and values.
    Query(DenseOutputSource<'a>),
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
    /// `None` in post-norm layers, as in [`DecodeAttentionWeights`].
    pub input_norm: Option<&'a CudaTensor>,
    pub qkv: DenseQkvSource<'a>,
    pub query_norm: Option<&'a CudaTensor>,
    pub key_norm: Option<&'a CudaTensor>,
    pub output: DenseOutputSource<'a>,
    pub post_attention_norm: Option<&'a CudaTensor>,
    pub sandwich: Option<DenseSandwichWeights<'a>>,
    pub feed_forward: DenseFeedForwardSource<'a>,
    pub per_layer_input: Option<PerLayerInputWeights<'a>>,
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
    input_norm: Option<CudaTensor>,
    qkv: DenseQkvOwned,
    query_norm: DeviceBuffer<bf16>,
    key_norm: DeviceBuffer<bf16>,
    output: DenseOutputOwned,
    post_attention_norm: Option<CudaTensor>,
    sandwich: Option<DenseSandwichOwned>,
    feed_forward: FeedForwardOwned,
    per_layer_input: Option<PerLayerInputOwned>,
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

impl DenseWeights {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        source: DenseWeightSource<'_>,
    ) -> Result<Self> {
        let [query_norm, key_norm] = qk_norms(backend, config.attention, source)?;
        Ok(Self {
            input_norm: source.input_norm.cloned(),
            qkv: source.qkv.into(),
            query_norm,
            key_norm,
            output: DenseOutputOwned::new(backend, source.output)?,
            post_attention_norm: source.post_attention_norm.cloned(),
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
            per_layer_input: source.per_layer_input.map(PerLayerInputOwned::new),
        })
    }

    pub(super) fn borrow(&self) -> DenseSwiGluWeights<'_> {
        DenseSwiGluWeights {
            attention: DecodeAttentionWeights {
                input_norm: self.input_norm.as_ref(),
                qkv: self.qkv.borrow(),
                query_norm: &self.query_norm,
                key_norm: &self.key_norm,
                output: self.output.borrow(),
            },
            post_attention_norm: self.post_attention_norm.as_ref(),
            sandwich: self.sandwich.as_ref().map(DenseSandwichOwned::borrow),
            feed_forward: match &self.feed_forward {
                FeedForwardOwned::Gated { gate_up, down } => DenseFeedForwardWeights::Gated {
                    gate_up: gate_up.borrow(),
                    down: down.borrow(),
                },
                FeedForwardOwned::Routed(weights) => DenseFeedForwardWeights::Routed(weights),
            },
            per_layer_input: self.per_layer_input.as_ref().map(PerLayerInputOwned::borrow),
        }
    }
}
