pub use runtime::scheduler::PrefillRefillPolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillAdmissionPolicy {
    #[cfg_attr(
        not(feature = "metal"),
        allow(dead_code, reason = "uniform admission is selected by Metal")
    )]
    Uniform,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "short-prompt admission is a CUDA mixed-attention policy")
    )]
    ShortPrompt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveDepthPolicy {
    Bounded,
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "unbounded waves remain exercised by scheduling policy tests")
    )]
    Unbounded,
}

impl WaveDepthPolicy {
    #[must_use]
    pub const fn is_bounded(self) -> bool {
        matches!(self, Self::Bounded)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeInterleaving {
    Interleaved,
    CohortSerial,
}

impl DecodeInterleaving {
    #[must_use]
    pub const fn allows_interleaving(self) -> bool {
        matches!(self, Self::Interleaved)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewDecodePolicy {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "immediate decode admission is selected by CUDA")
    )]
    Immediate,
    #[cfg_attr(
        not(any(feature = "metal", test)),
        allow(dead_code, reason = "deferred decode admission is selected by Metal")
    )]
    DeferDuringPrefill,
}

impl NewDecodePolicy {
    #[must_use]
    pub const fn defers_during_prefill(self) -> bool {
        matches!(self, Self::DeferDuringPrefill)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixReplayPolicy {
    Off,
    Replay {
        fallback_tokens: usize,
        checkpoint_tokens: Option<usize>,
        completion_slack_tokens: usize,
    },
}

impl PrefixReplayPolicy {
    #[must_use]
    pub const fn fallback_tokens(self) -> Option<usize> {
        match self {
            Self::Off => None,
            Self::Replay { fallback_tokens, .. } => Some(fallback_tokens),
        }
    }

    #[must_use]
    pub const fn checkpoint_tokens(self) -> Option<usize> {
        match self {
            Self::Off => None,
            Self::Replay { checkpoint_tokens, .. } => checkpoint_tokens,
        }
    }

    #[must_use]
    pub const fn completion_slack_tokens(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Replay { completion_slack_tokens, .. } => completion_slack_tokens,
        }
    }
}

#[derive(Clone, Copy)]
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
    pub wave_depth: WaveDepthPolicy,
    pub prefix_replay: PrefixReplayPolicy,
    pub new_decode: NewDecodePolicy,
    pub interleaving: DecodeInterleaving,
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
                "short_prompt refill requires a CUDA mixed-attention model with combined steps"
                    .into(),
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
            super::EngineInner::Unavailable(never) => match *never {},
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
            self.interleaving = DecodeInterleaving::CohortSerial;
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
            wave_depth: WaveDepthPolicy::Bounded,
            prefix_replay: PrefixReplayPolicy::Off,
            new_decode: NewDecodePolicy::Immediate,
            interleaving: DecodeInterleaving::Interleaved,
        };
        let default = profile.with_decode_policy(PrefillDecodePolicy::BackendDefault);
        assert_eq!(default.interleaving, DecodeInterleaving::Interleaved);
        assert_eq!(default.max_prefill_cohort_tokens, usize::MAX);
        let grouped = profile.with_decode_policy(PrefillDecodePolicy::CompleteCohort);
        assert_eq!(grouped.interleaving, DecodeInterleaving::CohortSerial);
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
