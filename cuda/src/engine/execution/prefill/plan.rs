pub(super) fn round_rows_from_pending(pending: &[bool], cursor: usize) -> Vec<usize> {
    (0..pending.len())
        .map(|offset| (cursor + offset) % pending.len())
        .filter(|row| pending[*row])
        .collect()
}

/// A prompt remainder this short would cost a memory-bound forward pass of
/// its own, far more than its tokens cost inside the chunk before it.
pub(super) const ABSORBED_REMAINDER_TOKENS: usize = 64;

pub(super) const fn valid_chunk(count: usize, remaining: usize, budget: usize) -> bool {
    count > 0
        && count <= remaining
        && (count <= budget || (count == remaining && count - budget <= ABSORBED_REMAINDER_TOKENS))
}

/// Tokens a chunk's start is aligned to: K/V blocks, so a fair share does
/// not leave a row's next chunk straddling one.
pub(super) const CHUNK_ALIGNMENT_TOKENS: usize = 16;

/// A row's chunk within `limit`: the whole remainder when it fits, else the
/// limit rounded down to the block alignment (at least one block).
pub(super) const fn aligned_chunk(limit: usize, remaining: usize) -> usize {
    if limit >= remaining {
        remaining
    } else {
        let aligned = limit / CHUNK_ALIGNMENT_TOKENS * CHUNK_ALIGNMENT_TOKENS;
        if aligned == 0 {
            limit
        } else {
            aligned
        }
    }
}

/// Extends a budget-limited chunk over the rest of the prompt when only a
/// short remainder would be left. Checkpoint boundaries still end a chunk.
pub(super) const fn absorb_remainder(limit: usize, remaining: usize, boundary: usize) -> usize {
    if limit > 0
        && limit < remaining
        && remaining - limit <= ABSORBED_REMAINDER_TOKENS
        && boundary >= remaining
    {
        remaining
    } else {
        limit
    }
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
