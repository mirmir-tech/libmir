use super::{DirectFp8ScaleDType, DirectFp8WeightScale};
use crate::ExecutionPhase;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(super) enum QuantizedProfileFormat {
    Affine {
        group_size: usize,
        bits: usize,
    },
    MxFp8,
    DirectFp8DynamicE4M3OutputChannel {
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    },
    DirectFp8StaticE4M3 {
        weight_scale: DirectFp8WeightScale,
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    },
    DirectFp8Bf16E5M2WeightOnly {
        bias: bool,
    },
    NvFp4Bf16WeightOnly,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(in crate::backend) struct QuantizedProfileRequest {
    pub(super) phase: ExecutionPhase,
    pub(super) tokens: usize,
    pub(super) input_features: usize,
    pub(super) output_features: usize,
    pub(super) format: QuantizedProfileFormat,
}

impl QuantizedProfileRequest {
    #[must_use]
    pub(in crate::backend) const fn tokens(self) -> usize {
        self.tokens
    }

    pub(in crate::backend) const fn affine(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        group_size: usize,
        bits: usize,
    ) -> Self {
        Self::new(
            tokens,
            input_features,
            output_features,
            QuantizedProfileFormat::Affine { group_size, bits },
        )
    }

    pub(in crate::backend) const fn mxfp8(
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Self {
        Self::new(tokens, input_features, output_features, QuantizedProfileFormat::MxFp8)
    }

    pub(in crate::backend) const fn direct_fp8_dynamic_e4m3(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    ) -> Self {
        Self::new(
            tokens,
            input_features,
            output_features,
            QuantizedProfileFormat::DirectFp8DynamicE4M3OutputChannel { scale_dtype, bias },
        )
    }

    pub(in crate::backend) const fn direct_fp8_static_e4m3(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        weight_scale: DirectFp8WeightScale,
        scale_dtype: DirectFp8ScaleDType,
        bias: bool,
    ) -> Self {
        Self::new(
            tokens,
            input_features,
            output_features,
            QuantizedProfileFormat::DirectFp8StaticE4M3 { weight_scale, scale_dtype, bias },
        )
    }

    pub(in crate::backend) const fn direct_fp8_bf16_e5m2_weight_only(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        bias: bool,
    ) -> Self {
        Self::new(
            tokens,
            input_features,
            output_features,
            QuantizedProfileFormat::DirectFp8Bf16E5M2WeightOnly { bias },
        )
    }

    pub(in crate::backend) const fn nvfp4_bf16_weight_only(
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Self {
        Self::new(
            tokens,
            input_features,
            output_features,
            QuantizedProfileFormat::NvFp4Bf16WeightOnly,
        )
    }

    const fn new(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        format: QuantizedProfileFormat,
    ) -> Self {
        Self {
            phase: if tokens == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            tokens,
            input_features,
            output_features,
            format,
        }
    }
}
