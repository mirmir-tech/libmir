use super::{
    ClampedMoeStorage, MoeProfileFormat, MoeProfileRequest, MxFp4MoeStorage, MxFp8MoeStorage,
};
use crate::{ExecutionPhase, GatedActivation, MoePlanRequest};

const CLAMPED_PREFILL_TOKEN_QUANTUM: usize = 256;

impl MoeProfileRequest {
    pub(in crate::backend) const fn nvfp4(
        plan: MoePlanRequest,
        activation: GatedActivation,
        weight_only: bool,
    ) -> Self {
        Self {
            phase: plan.phase,
            tokens: plan.tokens,
            experts: plan.experts,
            top_k: plan.top_k,
            hidden_features: plan.hidden_features,
            intermediate_features: plan.intermediate_features,
            format: MoeProfileFormat::NvFp4 { activation, weight_only },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) const fn affine(
        phase: ExecutionPhase,
        tokens: usize,
        experts: usize,
        top_k: usize,
        hidden_features: usize,
        intermediate_features: usize,
        group_size: usize,
        bits: usize,
        activation: GatedActivation,
    ) -> Self {
        Self {
            phase,
            tokens,
            experts,
            top_k,
            hidden_features,
            intermediate_features,
            format: MoeProfileFormat::Affine { group_size, bits, activation },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) const fn clamped(
        phase: ExecutionPhase,
        tokens: usize,
        experts: usize,
        top_k: usize,
        hidden_features: usize,
        intermediate_features: usize,
        storage: ClampedMoeStorage,
    ) -> Self {
        Self {
            phase,
            tokens: clamped_profile_tokens(phase, tokens),
            experts,
            top_k,
            hidden_features,
            intermediate_features,
            format: MoeProfileFormat::Clamped { storage },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) const fn mxfp4(
        phase: ExecutionPhase,
        tokens: usize,
        experts: usize,
        top_k: usize,
        hidden_features: usize,
        intermediate_features: usize,
        storage: MxFp4MoeStorage,
        activation: GatedActivation,
    ) -> Self {
        Self::block(
            phase,
            tokens,
            experts,
            top_k,
            hidden_features,
            intermediate_features,
            MoeProfileFormat::MxFp4 { storage, activation },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) const fn mxfp8(
        phase: ExecutionPhase,
        tokens: usize,
        experts: usize,
        top_k: usize,
        hidden_features: usize,
        intermediate_features: usize,
        storage: MxFp8MoeStorage,
        bias: bool,
        activation: GatedActivation,
    ) -> Self {
        Self::block(
            phase,
            tokens,
            experts,
            top_k,
            hidden_features,
            intermediate_features,
            MoeProfileFormat::MxFp8 { storage, bias, activation },
        )
    }

    #[allow(clippy::too_many_arguments)]
    const fn block(
        phase: ExecutionPhase,
        tokens: usize,
        experts: usize,
        top_k: usize,
        hidden_features: usize,
        intermediate_features: usize,
        format: MoeProfileFormat,
    ) -> Self {
        Self {
            phase,
            tokens,
            experts,
            top_k,
            hidden_features,
            intermediate_features,
            format,
        }
    }
}

const fn clamped_profile_tokens(phase: ExecutionPhase, tokens: usize) -> usize {
    if matches!(phase, ExecutionPhase::Prefill) {
        tokens.saturating_add(CLAMPED_PREFILL_TOKEN_QUANTUM - 1) / CLAMPED_PREFILL_TOKEN_QUANTUM
            * CLAMPED_PREFILL_TOKEN_QUANTUM
    } else {
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamped_prefill_profiles_share_nearby_token_geometry() {
        assert_eq!(
            clamped_profile_tokens(ExecutionPhase::Prefill, 1_968),
            clamped_profile_tokens(ExecutionPhase::Prefill, 2_032)
        );
        assert_ne!(
            clamped_profile_tokens(ExecutionPhase::Prefill, 1_792),
            clamped_profile_tokens(ExecutionPhase::Prefill, 2_032)
        );
        assert_eq!(clamped_profile_tokens(ExecutionPhase::Decode, 31), 31);
    }

    #[test]
    fn clamped_geometry_ignores_phase_and_tokens_but_not_storage() {
        let native = MoeProfileRequest::clamped(
            ExecutionPhase::Prefill,
            2_032,
            32,
            4,
            2_880,
            2_880,
            ClampedMoeStorage::Native,
        );
        let decode = MoeProfileRequest::clamped(
            ExecutionPhase::Decode,
            1,
            32,
            4,
            2_880,
            2_880,
            ClampedMoeStorage::Native,
        );
        let mlx = MoeProfileRequest {
            format: MoeProfileFormat::Clamped { storage: ClampedMoeStorage::Mlx },
            ..decode
        };
        assert!(native.same_clamped_geometry(decode));
        assert!(!native.same_clamped_geometry(mlx));
    }
}
