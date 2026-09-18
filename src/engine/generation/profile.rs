pub use runtime::scheduler::PrefillRefillPolicy;

#[derive(Clone, Copy)]
pub enum PrefillAdmissionPolicy {
    #[cfg_attr(
        not(feature = "metal"),
        allow(dead_code, reason = "uniform admission is selected by Metal")
    )]
    Uniform,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "short-prompt admission is a CUDA dense mixed-attention policy")
    )]
    ShortPrompt,
}

#[derive(Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "backend scheduling capabilities are independent binary contracts"
)]
pub struct PrefillExecutionProfile {
    pub refill: PrefillRefillPolicy,
    pub admission: PrefillAdmissionPolicy,
    pub chunk_tokens: usize,
    pub completion_round_tokens: usize,
    pub max_prefill_wave_rows: usize,
    pub max_prefill_wave_tokens: usize,
    pub max_prefill_cohort_tokens: usize,
    pub block_tokens: usize,
    pub resident_token_slots: usize,
    pub limit_deep_prefill_waves: bool,
    pub cached_prefix_replay_tokens: Option<usize>,
    pub cached_prefix_checkpoint_replay_tokens: Option<usize>,
    pub cached_prefix_completion_slack_tokens: usize,
    pub defer_new_decode: bool,
    pub interleave_prefill_decode: bool,
}

impl super::Engine {
    pub(super) fn prefill_admission_policy(
        &self,
        model: &runtime::backend::ModelHandle,
        refill: PrefillRefillPolicy,
    ) -> crate::Result<PrefillAdmissionPolicy> {
        #[cfg(not(feature = "cuda"))]
        let _ = model;
        let admission = match &self.inner {
            #[cfg(feature = "cuda")]
            super::EngineInner::Cuda(cuda)
                if cuda.prefill_schedule(&model.id)?
                    == cuda::CudaPrefillSchedule::CompletionFirst =>
            {
                PrefillAdmissionPolicy::ShortPrompt
            },
            _ => PrefillAdmissionPolicy::Uniform,
        };
        if refill == PrefillRefillPolicy::ShortPrompt
            && !matches!(admission, PrefillAdmissionPolicy::ShortPrompt)
        {
            return Err(runtime::RuntimeError::Config(
                "short_prompt refill requires a CUDA dense mixed-attention model".into(),
            )
            .into());
        }
        Ok(admission)
    }

    #[cfg_attr(
        not(feature = "metal"),
        allow(
            clippy::unnecessary_wraps,
            reason = "only Metal refreshes fallible model memory limits"
        )
    )]
    pub(crate) fn refresh_prefill_memory_limits(
        &self,
        model: &runtime::backend::ModelHandle,
        profile: &mut PrefillExecutionProfile,
    ) -> crate::Result<()> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            super::EngineInner::Cuda(_) => {
                let _ = (model, profile);
            },
            #[cfg(feature = "metal")]
            super::EngineInner::Metal(metal) => {
                let schedule = metal.prefill_schedule(model)?;
                profile.max_prefill_wave_rows = schedule.max_wave_rows;
                profile.max_prefill_wave_tokens = schedule.max_wave_tokens;
                profile.max_prefill_cohort_tokens =
                    schedule.max_cohort_tokens.min(profile.resident_token_slots);
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            super::EngineInner::Unavailable => {
                let _ = (model, profile);
            },
        }
        Ok(())
    }
}

impl PrefillExecutionProfile {
    pub(crate) fn with_decode_policy(
        mut self,
        policy: runtime::scheduler::PrefillDecodePolicy,
    ) -> Self {
        if policy == runtime::scheduler::PrefillDecodePolicy::CompleteCohort {
            self.interleave_prefill_decode = false;
            self.max_prefill_cohort_tokens =
                self.max_prefill_cohort_tokens.min(self.resident_token_slots.max(1));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use runtime::scheduler::PrefillDecodePolicy;

    use super::*;

    #[test]
    fn completion_policy_is_opt_in_and_bounds_the_cohort_to_resident_slots() {
        let profile = PrefillExecutionProfile {
            refill: PrefillRefillPolicy::Closed,
            admission: PrefillAdmissionPolicy::Uniform,
            chunk_tokens: 1024,
            completion_round_tokens: 1024,
            max_prefill_wave_rows: usize::MAX,
            max_prefill_wave_tokens: usize::MAX,
            max_prefill_cohort_tokens: usize::MAX,
            block_tokens: 16,
            resident_token_slots: 81920,
            limit_deep_prefill_waves: true,
            cached_prefix_replay_tokens: None,
            cached_prefix_checkpoint_replay_tokens: None,
            cached_prefix_completion_slack_tokens: 0,
            defer_new_decode: false,
            interleave_prefill_decode: true,
        };
        let default = profile.with_decode_policy(PrefillDecodePolicy::BackendDefault);
        assert!(default.interleave_prefill_decode);
        assert_eq!(default.max_prefill_cohort_tokens, usize::MAX);
        let grouped = profile.with_decode_policy(PrefillDecodePolicy::CompleteCohort);
        assert!(!grouped.interleave_prefill_decode);
        assert_eq!(grouped.max_prefill_cohort_tokens, 81920);
        let smaller = PrefillExecutionProfile {
            max_prefill_cohort_tokens: 1024,
            ..profile
        };
        assert_eq!(
            smaller
                .with_decode_policy(PrefillDecodePolicy::CompleteCohort)
                .max_prefill_cohort_tokens,
            1024
        );
    }
}
