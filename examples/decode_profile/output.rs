use std::env;

use cuda::{CudaKernelAdmission, CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy};

#[derive(Clone, Copy)]
pub(super) enum OutputMode {
    Bf16,
    Bf16Experimental,
    AutoRefined,
    Fp8Vectorized,
    Fp8Residual,
    Fp8BlockVectorized,
    Fp8BlockRefined,
}

impl OutputMode {
    pub(super) fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        match env::var("MIRMIR_CUDA_OUTPUT_HEAD").as_deref().unwrap_or("bf16") {
            "bf16" => Ok(Self::Bf16),
            "bf16-experimental" => Ok(Self::Bf16Experimental),
            "auto-refined" => Ok(Self::AutoRefined),
            "fp8-vectorized" => Ok(Self::Fp8Vectorized),
            "fp8-residual" => Ok(Self::Fp8Residual),
            "fp8-block-vectorized" => Ok(Self::Fp8BlockVectorized),
            "fp8-block-refined" => Ok(Self::Fp8BlockRefined),
            _ => Err("unsupported CUDA output-head mode".into()),
        }
    }

    pub(super) fn configure(self, policy: &mut CudaPlanningPolicy) {
        policy.output_head = match self {
            Self::Bf16 | Self::Bf16Experimental => CudaOutputHeadPolicy::Bf16,
            Self::AutoRefined => CudaOutputHeadPolicy::Auto,
            Self::Fp8Vectorized => CudaOutputHeadPolicy::Fp8Vectorized,
            Self::Fp8Residual => CudaOutputHeadPolicy::Fp8Residual,
            Self::Fp8BlockVectorized => CudaOutputHeadPolicy::Fp8BlockVectorized,
            Self::Fp8BlockRefined => CudaOutputHeadPolicy::Fp8BlockRefined,
        };
        if !matches!(self, Self::Bf16) {
            policy.numerical = CudaNumericalPolicy::Throughput;
            policy.admission = CudaKernelAdmission::Experimental;
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Bf16 => " output=bf16",
            Self::Bf16Experimental => " output=bf16-experimental",
            Self::AutoRefined => " output=auto-refined",
            Self::Fp8Vectorized => " output=fp8-vectorized",
            Self::Fp8Residual => " output=fp8-residual",
            Self::Fp8BlockVectorized => " output=fp8-block-vectorized",
            Self::Fp8BlockRefined => " output=fp8-block-refined",
        }
    }
}
