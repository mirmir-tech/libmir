use models::semantic::{FeedForwardSpec, MixerSpec, SemanticModelSpec};

const DEFAULT_PREFILL_STEP: usize = 512;
const LARGE_PREFILL_STEP: usize = 2_048;
const MAX_PREFILL_TOKEN_PAIRS: usize = 8 * 1_024 * 1_024;
const MAX_PREFILL_CHUNK_TOKENS: usize = 512;
const NATIVE_PAGED_TAIL_CONTEXT: usize = 16 * 1_024;
const NATIVE_PAGED_TAIL_TOKENS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefillArchitecture {
    linear_attention: bool,
    routed_experts: bool,
}

pub(super) fn prefill_step(spec: &SemanticModelSpec, configured: Option<usize>) -> usize {
    configured.filter(|step| *step > 0).unwrap_or_else(|| {
        default_prefill_step(PrefillArchitecture {
            linear_attention: spec
                .decoder
                .layers
                .iter()
                .any(|layer| matches!(layer.mixer, MixerSpec::LinearAttention(_))),
            routed_experts: spec.decoder.layers.iter().any(|layer| {
                matches!(
                    &layer.feed_forward,
                    FeedForwardSpec::Routed { .. } | FeedForwardSpec::DenseAndRouted { .. }
                )
            }),
        })
    })
}

pub(super) fn bounded_prefill_step(configured: usize, position: usize, remaining: usize) -> usize {
    if position >= NATIVE_PAGED_TAIL_CONTEXT && remaining <= NATIVE_PAGED_TAIL_TOKENS {
        return remaining.min(1);
    }
    let context = position.saturating_add(1);
    let workspace_bound = (MAX_PREFILL_TOKEN_PAIRS / context).max(1);
    remaining.min(configured).min(MAX_PREFILL_CHUNK_TOKENS).min(workspace_bound)
}

const fn default_prefill_step(architecture: PrefillArchitecture) -> usize {
    if architecture.linear_attention || architecture.routed_experts {
        LARGE_PREFILL_STEP
    } else {
        DEFAULT_PREFILL_STEP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enlarges_prefill_for_linear_attention_and_routed_experts() {
        let dense = PrefillArchitecture {
            linear_attention: false,
            routed_experts: false,
        };
        let linear = PrefillArchitecture {
            linear_attention: true,
            routed_experts: false,
        };
        let routed = PrefillArchitecture {
            linear_attention: false,
            routed_experts: true,
        };
        assert_eq!(default_prefill_step(dense), 512);
        assert_eq!(default_prefill_step(linear), 2_048);
        assert_eq!(default_prefill_step(routed), 2_048);
    }

    #[test]
    fn bounds_long_context_chunks_by_attention_work() {
        assert_eq!(bounded_prefill_step(2_048, 0, 4_096), 512);
        assert_eq!(bounded_prefill_step(2_048, 16_384, 4_096), 511);
        assert_eq!(bounded_prefill_step(2_048, 32_768, 4_096), 255);
        assert_eq!(bounded_prefill_step(2_048, 32_767, 2), 1);
        assert_eq!(bounded_prefill_step(2_048, usize::MAX, 4_096), 1);
    }
}
