use mircuda::{DeviceBuffer, bf16};

use super::super::{DenseDownWeight, DenseGateUpWeights, DenseSwiGluConfig, DenseSwiGluWeights};
use crate::{
    Bf16LinearPackWeights, Bf16LinearPairWeights, CudaBackend, CudaTensor,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeQkvWeights, NvFp4LinearWeight,
    Result,
};

mod norm;

use norm::norm_buffer;

#[derive(Clone, Copy)]
pub enum DenseQkvSource<'a> {
    Bf16(&'a Bf16LinearPackWeights<3>),
    NvFp4([&'a NvFp4LinearWeight; 3]),
}

#[derive(Clone, Copy)]
pub enum DenseOutputSource<'a> {
    Bf16(&'a CudaTensor),
    NvFp4(&'a NvFp4LinearWeight),
}

#[derive(Clone, Copy)]
pub enum DenseGateUpSource<'a> {
    Bf16(&'a Bf16LinearPairWeights),
    NvFp4 {
        gate: &'a NvFp4LinearWeight,
        up: &'a NvFp4LinearWeight,
    },
}

#[derive(Clone, Copy)]
pub enum DenseDownSource<'a> {
    Bf16(&'a CudaTensor),
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
    pub gate_up: DenseGateUpSource<'a>,
    pub down: DenseDownSource<'a>,
}

#[derive(Clone)]
pub(super) struct DenseWeights {
    input_norm: CudaTensor,
    qkv: DenseQkvOwned,
    query_norm: DeviceBuffer<bf16>,
    key_norm: DeviceBuffer<bf16>,
    output: DenseOutputOwned,
    post_attention_norm: CudaTensor,
    gate_up: DenseGateUpOwned,
    down: DenseDownOwned,
}

#[derive(Clone)]
enum DenseQkvOwned {
    Bf16(Bf16LinearPackWeights<3>),
    NvFp4([NvFp4LinearWeight; 3]),
}

#[derive(Clone)]
enum DenseOutputOwned {
    Bf16(CudaTensor),
    NvFp4(NvFp4LinearWeight),
}

#[derive(Clone)]
enum DenseGateUpOwned {
    Bf16(Bf16LinearPairWeights),
    NvFp4 {
        gate: NvFp4LinearWeight,
        up: NvFp4LinearWeight,
    },
}

#[derive(Clone)]
enum DenseDownOwned {
    Bf16(CudaTensor),
    NvFp4(NvFp4LinearWeight),
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
            output: source.output.into(),
            post_attention_norm: source.post_attention_norm.clone(),
            gate_up: source.gate_up.into(),
            down: source.down.into(),
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
            gate_up: self.gate_up.borrow(),
            down: self.down.borrow(),
        }
    }
}

impl DenseQkvOwned {
    fn borrow(&self) -> DecodeQkvWeights<'_> {
        match self {
            Self::Bf16(weights) => DecodeQkvWeights::Bf16(weights),
            Self::NvFp4(weights) => {
                DecodeQkvWeights::NvFp4([&weights[0], &weights[1], &weights[2]])
            },
        }
    }
}

impl DenseOutputOwned {
    fn borrow(&self) -> DecodeAttentionOutputWeight<'_> {
        match self {
            Self::Bf16(weight) => DecodeAttentionOutputWeight::Bf16(weight),
            Self::NvFp4(weight) => DecodeAttentionOutputWeight::NvFp4(weight),
        }
    }
}

impl DenseGateUpOwned {
    fn borrow(&self) -> DenseGateUpWeights<'_> {
        match self {
            Self::Bf16(weights) => DenseGateUpWeights::Bf16(weights),
            Self::NvFp4 { gate, up } => DenseGateUpWeights::NvFp4 { gate, up },
        }
    }
}

impl DenseDownOwned {
    fn borrow(&self) -> DenseDownWeight<'_> {
        match self {
            Self::Bf16(weight) => DenseDownWeight::Bf16(weight),
            Self::NvFp4(weight) => DenseDownWeight::NvFp4(weight),
        }
    }
}

impl From<DenseQkvSource<'_>> for DenseQkvOwned {
    fn from(source: DenseQkvSource<'_>) -> Self {
        match source {
            DenseQkvSource::Bf16(weights) => Self::Bf16(weights.clone()),
            DenseQkvSource::NvFp4(weights) => Self::NvFp4(weights.map(Clone::clone)),
        }
    }
}

impl From<DenseOutputSource<'_>> for DenseOutputOwned {
    fn from(source: DenseOutputSource<'_>) -> Self {
        match source {
            DenseOutputSource::Bf16(weight) => Self::Bf16(weight.clone()),
            DenseOutputSource::NvFp4(weight) => Self::NvFp4(weight.clone()),
        }
    }
}

impl From<DenseGateUpSource<'_>> for DenseGateUpOwned {
    fn from(source: DenseGateUpSource<'_>) -> Self {
        match source {
            DenseGateUpSource::Bf16(weights) => Self::Bf16(weights.clone()),
            DenseGateUpSource::NvFp4 { gate, up } => {
                Self::NvFp4 { gate: gate.clone(), up: up.clone() }
            },
        }
    }
}

impl From<DenseDownSource<'_>> for DenseDownOwned {
    fn from(source: DenseDownSource<'_>) -> Self {
        match source {
            DenseDownSource::Bf16(weight) => Self::Bf16(weight.clone()),
            DenseDownSource::NvFp4(weight) => Self::NvFp4(weight.clone()),
        }
    }
}
