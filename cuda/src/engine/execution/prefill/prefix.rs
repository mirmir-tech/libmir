use std::num::NonZeroUsize;

use runtime::{backend::PrefillRequest, kv::reusable_prefix_tokens};

use crate::{Result, engine::model::ModelExecution};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PrefixReuse {
    #[default]
    None,
    BlockFallback {
        tokens: NonZeroUsize,
    },
    CheckpointRestored {
        tokens: NonZeroUsize,
    },
}

impl PrefixReuse {
    pub(super) const fn tokens(self) -> usize {
        match self {
            Self::None => 0,
            Self::BlockFallback { tokens } | Self::CheckpointRestored { tokens } => tokens.get(),
        }
    }

    pub(super) const fn is_checkpoint_restored(self) -> bool {
        matches!(self, Self::CheckpointRestored { .. })
    }
}

pub(super) fn prepare(
    execution: &mut ModelExecution,
    request: &PrefillRequest,
) -> Result<PrefixReuse> {
    let ModelExecution::Generation(generation) = execution else {
        return Ok(PrefixReuse::None);
    };
    let Some(replay_tokens) = generation.prefix_replay_tokens() else {
        return Ok(PrefixReuse::None);
    };
    let fallback = reusable_prefix_tokens(
        request.cached_tokens,
        request.prompt_tokens.len(),
        request.block_table.block_size(),
        replay_tokens,
    );
    let maximum = reusable_prefix_tokens(
        request.cached_tokens,
        request.prompt_tokens.len(),
        request.block_table.block_size(),
        0,
    );
    let restored = generation.restore_prefix(request, fallback, maximum)?;
    Ok(resolve(fallback, restored))
}

fn resolve(fallback: usize, restored: Option<usize>) -> PrefixReuse {
    if let Some(tokens) = restored.and_then(NonZeroUsize::new) {
        return PrefixReuse::CheckpointRestored { tokens };
    }
    NonZeroUsize::new(fallback)
        .map_or(PrefixReuse::None, |tokens| PrefixReuse::BlockFallback { tokens })
}

#[cfg(test)]
mod tests {
    use super::{PrefixReuse, resolve};

    #[test]
    fn distinguishes_checkpoint_restore_from_correctness_fallback() {
        let none = resolve(0, None);
        assert_eq!(none, PrefixReuse::None);
        assert_eq!(none.tokens(), 0);
        assert!(!none.is_checkpoint_restored());

        let fallback = resolve(6_667, None);
        assert_eq!(fallback.tokens(), 6_667);
        assert!(!fallback.is_checkpoint_restored());

        let restored = resolve(6_667, Some(8_176));
        assert_eq!(restored.tokens(), 8_176);
        assert!(restored.is_checkpoint_restored());
    }
}
