use crate::{
    AffineQuantizedWeight, Bf16LinearPairWeights, BlockFp8LinearWeight, CudaTensor,
    DenseDownWeight, DenseGateUpWeights, DenseSwiGluWeights, DirectFp8CheckpointWeight,
    Fp8ResidualLinearWeight, MxFp4CheckpointWeight, MxFp8CheckpointWeight, NvFp4LinearWeight,
    PackedIntegerWeight, Result, backend::attention::graph::CapturedAttentionWeights,
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
    Affine {
        gate: AffineQuantizedWeight,
        up: AffineQuantizedWeight,
    },
    Bf16(Bf16LinearPairWeights),
    DirectFp8 {
        gate: DirectFp8CheckpointWeight,
        up: DirectFp8CheckpointWeight,
    },
    MxFp4 {
        gate: MxFp4CheckpointWeight,
        up: MxFp4CheckpointWeight,
    },
    MxFp8 {
        gate: MxFp8CheckpointWeight,
        up: MxFp8CheckpointWeight,
    },
    PackedInteger {
        gate: PackedIntegerWeight,
        up: PackedIntegerWeight,
    },
    NvFp4 {
        gate: NvFp4LinearWeight,
        up: NvFp4LinearWeight,
    },
    BlockFp8 {
        exact: Bf16LinearPairWeights,
        quantized: BlockFp8LinearWeight,
    },
    Fp8Int4 {
        exact: Bf16LinearPairWeights,
        quantized: Fp8ResidualLinearWeight,
    },
}

#[derive(Clone)]
enum CapturedDown {
    Affine(AffineQuantizedWeight),
    Bf16(CudaTensor),
    DirectFp8(DirectFp8CheckpointWeight),
    MxFp4(MxFp4CheckpointWeight),
    MxFp8(MxFp8CheckpointWeight),
    PackedInteger(PackedIntegerWeight),
    NvFp4(NvFp4LinearWeight),
    BlockFp8 {
        exact: CudaTensor,
        quantized: BlockFp8LinearWeight,
    },
    Fp8Int4 {
        exact: CudaTensor,
        quantized: Fp8ResidualLinearWeight,
    },
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
            DenseGateUpWeights::Affine { gate, up } => {
                Self::Affine { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::Bf16(weights) => Self::Bf16(weights.clone()),
            DenseGateUpWeights::DirectFp8 { gate, up } => {
                Self::DirectFp8 { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::MxFp4 { gate, up } => {
                Self::MxFp4 { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::MxFp8 { gate, up } => {
                Self::MxFp8 { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::PackedInteger { gate, up } => {
                Self::PackedInteger { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::NvFp4 { gate, up } => {
                Self::NvFp4 { gate: gate.clone(), up: up.clone() }
            },
            DenseGateUpWeights::BlockFp8 { exact, quantized } => Self::BlockFp8 {
                exact: exact.clone(),
                quantized: quantized.clone(),
            },
            DenseGateUpWeights::Fp8Int4 { exact, quantized } => Self::Fp8Int4 {
                exact: exact.clone(),
                quantized: quantized.clone(),
            },
        }
    }
}

impl CapturedGateUp {
    fn borrow(&self) -> DenseGateUpWeights<'_> {
        match self {
            Self::Affine { gate, up } => DenseGateUpWeights::Affine { gate, up },
            Self::Bf16(weights) => DenseGateUpWeights::Bf16(weights),
            Self::DirectFp8 { gate, up } => DenseGateUpWeights::DirectFp8 { gate, up },
            Self::MxFp4 { gate, up } => DenseGateUpWeights::MxFp4 { gate, up },
            Self::MxFp8 { gate, up } => DenseGateUpWeights::MxFp8 { gate, up },
            Self::PackedInteger { gate, up } => DenseGateUpWeights::PackedInteger { gate, up },
            Self::NvFp4 { gate, up } => DenseGateUpWeights::NvFp4 { gate, up },
            Self::BlockFp8 { exact, quantized } => {
                DenseGateUpWeights::BlockFp8 { exact, quantized }
            },
            Self::Fp8Int4 { exact, quantized } => DenseGateUpWeights::Fp8Int4 { exact, quantized },
        }
    }
}

impl From<DenseDownWeight<'_>> for CapturedDown {
    fn from(weight: DenseDownWeight<'_>) -> Self {
        match weight {
            DenseDownWeight::Affine(weight) => Self::Affine(weight.clone()),
            DenseDownWeight::Bf16(weight) => Self::Bf16(weight.clone()),
            DenseDownWeight::DirectFp8(weight) => Self::DirectFp8(weight.clone()),
            DenseDownWeight::MxFp4(weight) => Self::MxFp4(weight.clone()),
            DenseDownWeight::MxFp8(weight) => Self::MxFp8(weight.clone()),
            DenseDownWeight::PackedInteger(weight) => Self::PackedInteger(weight.clone()),
            DenseDownWeight::NvFp4(weight) => Self::NvFp4(weight.clone()),
            DenseDownWeight::BlockFp8 { exact, quantized } => Self::BlockFp8 {
                exact: exact.clone(),
                quantized: quantized.clone(),
            },
            DenseDownWeight::Fp8Int4 { exact, quantized } => Self::Fp8Int4 {
                exact: exact.clone(),
                quantized: quantized.clone(),
            },
        }
    }
}

impl CapturedDown {
    fn borrow(&self) -> DenseDownWeight<'_> {
        match self {
            Self::Affine(weight) => DenseDownWeight::Affine(weight),
            Self::Bf16(weight) => DenseDownWeight::Bf16(weight),
            Self::DirectFp8(weight) => DenseDownWeight::DirectFp8(weight),
            Self::MxFp4(weight) => DenseDownWeight::MxFp4(weight),
            Self::MxFp8(weight) => DenseDownWeight::MxFp8(weight),
            Self::PackedInteger(weight) => DenseDownWeight::PackedInteger(weight),
            Self::NvFp4(weight) => DenseDownWeight::NvFp4(weight),
            Self::BlockFp8 { exact, quantized } => DenseDownWeight::BlockFp8 { exact, quantized },
            Self::Fp8Int4 { exact, quantized } => DenseDownWeight::Fp8Int4 { exact, quantized },
        }
    }
}
