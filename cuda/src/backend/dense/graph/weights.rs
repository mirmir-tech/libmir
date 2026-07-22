use crate::{
    Bf16LinearPairWeights, CompressedInt8Weight, CudaTensor, DenseDownWeight, DenseGateUpWeights,
    DenseSwiGluWeights, NvFp4LinearWeight, Result,
    backend::attention::graph::CapturedAttentionWeights,
};

#[derive(Clone)]
pub(super) struct CapturedDenseWeights {
    attention: CapturedAttentionWeights,
    post_attention_norm: CudaTensor,
    gate_up: CapturedGateUp,
    down: CapturedDown,
}

#[derive(Clone)]
enum CapturedGateUp {
    Bf16(Bf16LinearPairWeights),
    Int8 {
        gate: CompressedInt8Weight,
        up: CompressedInt8Weight,
    },
    NvFp4 {
        gate: NvFp4LinearWeight,
        up: NvFp4LinearWeight,
    },
}

#[derive(Clone)]
enum CapturedDown {
    Bf16(CudaTensor),
    Int8(CompressedInt8Weight),
    NvFp4(NvFp4LinearWeight),
}

impl CapturedDenseWeights {
    pub(super) fn borrow(&self) -> DenseSwiGluWeights<'_> {
        DenseSwiGluWeights {
            attention: self.attention.borrow(),
            post_attention_norm: &self.post_attention_norm,
            gate_up: self.gate_up.borrow(),
            down: self.down.borrow(),
        }
    }
}

impl TryFrom<DenseSwiGluWeights<'_>> for CapturedDenseWeights {
    type Error = crate::Error;

    fn try_from(weights: DenseSwiGluWeights<'_>) -> Result<Self> {
        Ok(Self {
            attention: weights.attention.into(),
            post_attention_norm: weights.post_attention_norm.clone(),
            gate_up: weights.gate_up.into(),
            down: weights.down.into(),
        })
    }
}

impl From<DenseGateUpWeights<'_>> for CapturedGateUp {
    fn from(weights: DenseGateUpWeights<'_>) -> Self {
        match weights {
            DenseGateUpWeights::Bf16(weights) => Self::Bf16(weights.clone()),
            DenseGateUpWeights::Int8 { gate, up } => {
                Self::Int8 { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::NvFp4 { gate, up } => {
                Self::NvFp4 { gate: gate.clone(), up: up.clone() }
            },
        }
    }
}

impl CapturedGateUp {
    fn borrow(&self) -> DenseGateUpWeights<'_> {
        match self {
            Self::Bf16(weights) => DenseGateUpWeights::Bf16(weights),
            Self::Int8 { gate, up } => DenseGateUpWeights::Int8 { gate, up },
            Self::NvFp4 { gate, up } => DenseGateUpWeights::NvFp4 { gate, up },
        }
    }
}

impl From<DenseDownWeight<'_>> for CapturedDown {
    fn from(weight: DenseDownWeight<'_>) -> Self {
        match weight {
            DenseDownWeight::Bf16(weight) => Self::Bf16(weight.clone()),
            DenseDownWeight::Int8(weight) => Self::Int8(weight.clone()),
            DenseDownWeight::NvFp4(weight) => Self::NvFp4(weight.clone()),
        }
    }
}

impl CapturedDown {
    fn borrow(&self) -> DenseDownWeight<'_> {
        match self {
            Self::Bf16(weight) => DenseDownWeight::Bf16(weight),
            Self::Int8(weight) => DenseDownWeight::Int8(weight),
            Self::NvFp4(weight) => DenseDownWeight::NvFp4(weight),
        }
    }
}
