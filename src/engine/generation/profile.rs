#[derive(Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "backend scheduling capabilities are independent binary contracts"
)]
pub struct PrefillExecutionProfile {
    pub chunk_tokens: usize,
    pub completion_round_tokens: usize,
    pub max_prefill_wave_rows: usize,
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
