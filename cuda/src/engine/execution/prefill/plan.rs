pub(super) fn round_rows_from_pending(pending: &[bool], cursor: usize) -> Vec<usize> {
    (0..pending.len())
        .map(|offset| (cursor + offset) % pending.len())
        .filter(|row| pending[*row])
        .collect()
}

pub(super) const fn valid_chunk(count: usize, remaining: usize, budget: usize) -> bool {
    count > 0 && count <= remaining && count <= budget
}

pub(super) fn checkpoint_distance(
    consumed: usize,
    declared: &[usize],
    terminal: Option<usize>,
    alignment: Option<usize>,
) -> usize {
    let aligned = |checkpoint: &usize| {
        alignment.is_none_or(|alignment| alignment > 0 && checkpoint.is_multiple_of(alignment))
    };
    declared
        .iter()
        .filter(|checkpoint| **checkpoint > consumed)
        .find(|checkpoint| aligned(checkpoint))
        .copied()
        .into_iter()
        .chain(terminal.filter(|checkpoint| *checkpoint > consumed))
        .map(|checkpoint| checkpoint - consumed)
        .min()
        .unwrap_or(usize::MAX)
}

pub(super) const fn fair_chunk_budget(remaining_budget: usize, rows_left: usize) -> usize {
    remaining_budget.div_ceil(if rows_left == 0 {
        1
    } else {
        rows_left
    })
}

pub(super) const fn row_chunk_budget(
    remaining_budget: usize,
    rows_left: usize,
    completion_first: bool,
) -> usize {
    if completion_first {
        remaining_budget
    } else {
        fair_chunk_budget(remaining_budget, rows_left)
    }
}

pub(super) const fn context_chunk_budget(
    consumed_tokens: usize,
    rows: usize,
    token_budget: usize,
    interleaved_decode: bool,
    reused_prefix: bool,
    completion_first: bool,
) -> usize {
    if completion_first || consumed_tokens < 2_048 || (reused_prefix && !interleaved_decode) {
        usize::MAX
    } else if !interleaved_decode && rows.saturating_mul(2_048) <= token_budget {
        2_048
    } else {
        1_024
    }
}

pub(super) fn reusable_prefix_tokens(
    cached_tokens: usize,
    prompt_tokens: usize,
    block_size: Option<usize>,
    replay_tokens: usize,
) -> usize {
    let cached_tokens = cached_tokens.min(prompt_tokens);
    if replay_tokens > 0 {
        return cached_tokens.saturating_sub(replay_tokens);
    }
    if cached_tokens < prompt_tokens {
        return cached_tokens;
    }
    block_size
        .filter(|size| *size > 0)
        .map_or(0, |size| cached_tokens.saturating_sub(size))
}
