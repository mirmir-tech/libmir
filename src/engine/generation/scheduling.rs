use runtime::{backend::ModelHandle, kv::CacheConfig};

use super::{
    DecodeInterleaving, Engine, EngineInner, NewDecodePolicy, PrefillAdmissionPolicy,
    PrefillExecutionProfile, PrefillRefillPolicy, PrefixReplayPolicy, WaveDepthPolicy,
};
use crate::Result;
#[cfg(feature = "metal")]
const METAL_COMPLETION_ROUND_ROWS: usize = 16;
impl Engine {
    #[cfg_attr(
        not(feature = "metal"),
        allow(
            clippy::unnecessary_wraps,
            reason = "the Metal implementation performs a fallible model lookup"
        )
    )]
    pub(crate) fn generation_prefill_profile(
        &self,
        model: &ModelHandle,
        max_batch_tokens: usize,
        cache: CacheConfig,
        refill: PrefillRefillPolicy,
    ) -> Result<PrefillExecutionProfile> {
        #[cfg(not(feature = "cuda"))]
        let _ = max_batch_tokens;
        let admission = self.prefill_admission_policy(model, refill)?;
        let resident_token_slots =
            cache.block_size.saturating_mul(cache.block_count as usize).max(1);
        let chunk_tokens = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.prefill_chunk_tokens(),
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => metal.prefill_chunk_tokens(model)?,
        };
        let completion_round_tokens = self.completion_round_tokens(admission, max_batch_tokens);
        #[cfg(feature = "metal")]
        let metal_schedule = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => None,
            #[cfg(feature = "metal")]
            EngineInner::Metal(metal) => Some(metal.prefill_schedule(model)?),
        };
        let max_prefill_wave_rows = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => usize::MAX,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => metal_schedule.map_or(usize::MAX, |value| value.max_wave_rows),
        };
        let max_prefill_wave_tokens = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => usize::MAX,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => {
                metal_schedule.map_or(usize::MAX, |value| value.max_wave_tokens)
            },
        };
        let max_prefill_cohort_tokens = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => usize::MAX,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => metal_schedule
                .map_or(resident_token_slots, |value| value.max_cohort_tokens)
                .min(resident_token_slots),
        };
        let cached_prefix_admission = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => cuda.paged_prefix_admission(&model.id)?,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Some((0, 0, 0)),
        };
        let prefix_replay = cached_prefix_admission.map_or(
            PrefixReplayPolicy::Off,
            |(fallback_tokens, checkpoint_tokens, completion_slack_tokens)| {
                PrefixReplayPolicy::Replay {
                    fallback_tokens,
                    checkpoint_tokens: Some(checkpoint_tokens),
                    completion_slack_tokens,
                }
            },
        );
        let new_decode = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => NewDecodePolicy::Immediate,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => NewDecodePolicy::DeferDuringPrefill,
        };
        let interleaving = match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => DecodeInterleaving::Interleaved,
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => {
                if metal_schedule.is_none_or(|value| value.interleave_decode) {
                    DecodeInterleaving::Interleaved
                } else {
                    DecodeInterleaving::CohortSerial
                }
            },
        };
        Ok(PrefillExecutionProfile {
            refill,
            admission,
            chunk_tokens: chunk_tokens.max(1),
            completion_round_tokens: completion_round_tokens.max(1),
            max_prefill_wave_rows,
            max_prefill_wave_tokens,
            max_prefill_cohort_tokens,
            block_tokens: cache.block_size.max(1),
            resident_token_slots,
            wave_depth: WaveDepthPolicy::Bounded,
            prefix_replay,
            new_decode,
            interleaving,
        })
    }

    fn completion_round_tokens(
        &self,
        admission: PrefillAdmissionPolicy,
        max_batch_tokens: usize,
    ) -> usize {
        #[cfg(not(feature = "cuda"))]
        let _ = admission;
        match &self.inner {
            // A CUDA prefill wave holds every admitted row whatever its
            // length: rows share the step budget in rounds and pack into one
            // forward. Sizing waves by rows that finish inside one budget
            // left graph and sink-attention runtimes with one row per wave
            // for any prompt beyond the budget, so ten concurrent prompts
            // prefilled one after another while the others waited.
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(_) => {
                let _ = (admission, max_batch_tokens);
                1
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => max_batch_tokens.div_ceil(METAL_COMPLETION_ROUND_ROWS),
        }
    }
}
