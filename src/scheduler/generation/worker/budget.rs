use super::Worker;

pub(super) fn admission_replay_tokens(
    cached_tokens: usize,
    fallback_tokens: usize,
    checkpoint_tokens: Option<usize>,
) -> (usize, bool) {
    checkpoint_tokens
        .filter(|_| cached_tokens > fallback_tokens)
        .map_or((fallback_tokens, false), |tokens| (tokens, true))
}

pub(super) fn completion_work_tokens(
    tokens: usize,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_slack_tokens: usize,
) -> usize {
    if cached_tokens == 0 || cached_tokens >= prompt_tokens {
        tokens
    } else {
        tokens.saturating_sub(completion_slack_tokens).max(1)
    }
}

impl Worker {
    pub(super) fn prefill_work_tokens(&self, available: usize) -> usize {
        self.prefill
            .iter()
            .take(available)
            .map(|pending| self.prefill_completion_tokens(pending))
            .max()
            .unwrap_or(1)
    }

    pub(super) fn prefill_completion_tokens(
        &self,
        pending: &crate::scheduler::generation::PendingPrefill,
    ) -> usize {
        let Some(fallback_tokens) = self.prefill_profile.cached_prefix_replay_tokens else {
            return pending.request.prompt_tokens.len();
        };
        if !self.prefill_profile.limit_deep_prefill_waves {
            return pending.request.prompt_tokens.len();
        }
        let (replay_tokens, checkpoint) = admission_replay_tokens(
            pending.request.cached_tokens,
            fallback_tokens,
            self.prefill_profile.cached_prefix_checkpoint_replay_tokens,
        );
        let tokens = super::admission::pending_prefill_tokens(
            pending.request.prompt_tokens.len(),
            pending.request.cached_tokens,
            replay_tokens,
            self.prefill_profile.block_tokens,
        );
        completion_work_tokens(
            tokens,
            pending.request.prompt_tokens.len(),
            pending.request.cached_tokens,
            if checkpoint {
                self.prefill_profile.cached_prefix_completion_slack_tokens
            } else {
                0
            },
        )
    }

    pub(super) fn prefill_step_budget(&self, decode_rows: usize) -> usize {
        let cached_rows = self.active_prefill.as_ref().map_or(0, |active| {
            active
                .requests
                .iter()
                .filter(|row| {
                    row.request.cached_tokens < row.request.prompt_tokens.len()
                        && admission_replay_tokens(
                            row.request.cached_tokens,
                            self.prefill_profile.cached_prefix_replay_tokens.unwrap_or_default(),
                            self.prefill_profile.cached_prefix_checkpoint_replay_tokens,
                        )
                        .1
                })
                .count()
        });
        step_budget(
            self.config.max_batch_tokens,
            decode_rows,
            cached_rows,
            self.prefill_profile.cached_prefix_completion_slack_tokens,
        )
    }
}

fn step_budget(
    max_batch_tokens: usize,
    decode_rows: usize,
    cached_rows: usize,
    completion_slack_tokens: usize,
) -> usize {
    let base = max_batch_tokens.saturating_sub(decode_rows).max(1);
    if decode_rows > 0 {
        base
    } else {
        base.saturating_add(cached_rows.saturating_mul(completion_slack_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::{admission_replay_tokens, completion_work_tokens, step_budget};

    #[test]
    fn checkpoint_tail_admits_four_rows_with_one_block_of_slack() {
        assert_eq!(admission_replay_tokens(4_080, 1_525, Some(0)), (0, true));
        assert_eq!(completion_work_tokens(2_064, 6_144, 4_080, 16), 2_048);
        assert_eq!(step_budget(8_192, 0, 4, 16), 8_256);
    }

    #[test]
    fn misses_small_hits_and_interleaved_decode_do_not_receive_slack() {
        assert_eq!(admission_replay_tokens(80, 1_525, Some(0)), (1_525, false));
        assert_eq!(completion_work_tokens(6_144, 6_144, 0, 16), 6_144);
        assert_eq!(completion_work_tokens(16, 4_096, 4_096, 16), 16);
        assert_eq!(step_budget(8_192, 0, 0, 16), 8_192);
        assert_eq!(step_budget(8_192, 4, 4, 16), 8_188);
    }
}
