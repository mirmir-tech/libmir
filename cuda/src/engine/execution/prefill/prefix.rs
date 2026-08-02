use runtime::backend::PrefillRequest;

use super::plan;
use crate::{Result, engine::model::ModelExecution};

#[derive(Clone, Copy, Default)]
pub(super) struct PrefixReuse {
    pub(super) tokens: usize,
    pub(super) checkpoint_restored: bool,
}

pub(super) fn prepare(
    execution: &mut ModelExecution,
    request: &PrefillRequest,
) -> Result<PrefixReuse> {
    let ModelExecution::Generation(generation) = execution else {
        return Ok(PrefixReuse::default());
    };
    let Some(replay_tokens) = generation.prefix_replay_tokens() else {
        return Ok(PrefixReuse::default());
    };
    let fallback = plan::reusable_prefix_tokens(
        request.cached_tokens,
        request.prompt_tokens.len(),
        request.block_table.block_size(),
        replay_tokens,
    );
    let maximum = plan::reusable_prefix_tokens(
        request.cached_tokens,
        request.prompt_tokens.len(),
        request.block_table.block_size(),
        0,
    );
    let restored = generation.restore_prefix(request, fallback, maximum)?;
    Ok(resolve(fallback, restored))
}

fn resolve(fallback: usize, restored: Option<usize>) -> PrefixReuse {
    PrefixReuse {
        tokens: restored.unwrap_or(fallback),
        checkpoint_restored: restored.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::resolve;

    #[test]
    fn distinguishes_checkpoint_restore_from_correctness_fallback() {
        let fallback = resolve(6_667, None);
        assert_eq!(fallback.tokens, 6_667);
        assert!(!fallback.checkpoint_restored);

        let restored = resolve(6_667, Some(8_176));
        assert_eq!(restored.tokens, 8_176);
        assert!(restored.checkpoint_restored);
    }
}
