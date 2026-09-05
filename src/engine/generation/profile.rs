#[derive(Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "backend scheduling capabilities are independent binary contracts"
)]
pub struct PrefillExecutionProfile {
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
    pub collect_long_prefill_window: bool,
}

impl super::Engine {
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
