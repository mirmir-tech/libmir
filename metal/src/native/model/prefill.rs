use models::{
    semantic::{FeedForwardSpec, MixerSpec, SemanticModelSpec},
    weights::WeightBindingPlan,
};

const DEFAULT_PREFILL_STEP: usize = 512;
const LARGE_PREFILL_STEP: usize = 2_048;
// Keeps 2048-token chunks to 128k context; the earlier 8M bound shrank 32k
// chunks to 255 tokens and cost about a quarter of long-context prefill.
const MAX_PREFILL_TOKEN_PAIRS: usize = 256 * 1_024 * 1_024;
const MAX_PREFILL_CHUNK_TOKENS: usize = 2_048;
const NATIVE_PAGED_TAIL_CONTEXT: usize = 16 * 1_024;
const NATIVE_PAGED_TAIL_TOKENS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefillArchitecture {
    linear_attention: bool,
    routed_experts: bool,
    /// GGML blocks, which every prefill chunk dequantizes for MLX's matmul:
    /// Qwen3-8B `Q8_0` 1362 to 1615 and `Q4_K_M` 1394 to 1547 tok/s at 2048
    /// rather than 512-token chunks on M5 Pro, BF16 unchanged.
    dequantized_projections: bool,
}

pub(super) fn prefill_step(
    spec: &SemanticModelSpec,
    bindings: &WeightBindingPlan,
    configured: Option<usize>,
) -> usize {
    configured.filter(|step| *step > 0).unwrap_or_else(|| {
        default_prefill_step(PrefillArchitecture {
            dequantized_projections: bindings.uses_ggml(),
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
    bounded_step_with_limit(configured, position, remaining, MAX_PREFILL_CHUNK_TOKENS)
}

pub(super) fn bounded_step_with_limit(
    configured: usize,
    position: usize,
    remaining: usize,
    chunk_limit: usize,
) -> usize {
    if position >= NATIVE_PAGED_TAIL_CONTEXT && remaining <= NATIVE_PAGED_TAIL_TOKENS {
        return remaining.min(1);
    }
    let context = position.saturating_add(1);
    let workspace_bound = (MAX_PREFILL_TOKEN_PAIRS / context).max(1);
    remaining.min(configured).min(chunk_limit).min(workspace_bound)
}

const fn default_prefill_step(architecture: PrefillArchitecture) -> usize {
    if architecture.linear_attention
        || architecture.routed_experts
        || architecture.dequantized_projections
    {
        LARGE_PREFILL_STEP
    } else {
        DEFAULT_PREFILL_STEP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enlarges_prefill_for_linear_attention_routed_experts_and_ggml_blocks() {
        let dense = PrefillArchitecture {
            linear_attention: false,
            routed_experts: false,
            dequantized_projections: false,
        };
        let linear = PrefillArchitecture { linear_attention: true, ..dense };
        let ggml = PrefillArchitecture { dequantized_projections: true, ..dense };
        let routed = PrefillArchitecture { routed_experts: true, ..dense };
        assert_eq!(default_prefill_step(dense), 512);
        assert_eq!(default_prefill_step(linear), 2_048);
        assert_eq!(default_prefill_step(routed), 2_048);
        assert_eq!(default_prefill_step(ggml), 2_048);
    }

    #[test]
    fn larger_limits_preserve_workspace_and_long_context_tail_bounds() {
        assert_eq!(bounded_step_with_limit(2048, 0, 4096, 1024), 1024);
        assert_eq!(bounded_step_with_limit(2048, 0, 4096, 2048), 2048);
        assert_eq!(bounded_step_with_limit(512, 0, 4096, 2048), 512);
        assert_eq!(bounded_step_with_limit(2048, 16384, 4096, 2048), 2048);
        assert_eq!(bounded_step_with_limit(2048, 262_143, 4096, 2048), 1024);
        assert_eq!(bounded_step_with_limit(2048, 32767, 16, 2048), 1);
    }

    #[test]
    fn bounds_long_context_chunks_by_attention_work() {
        assert_eq!(bounded_prefill_step(2_048, 0, 4_096), 2_048);
        // MLX-LM keeps 2048-token steps; fused SDPA needs no score matrix.
        assert_eq!(bounded_prefill_step(2_048, 32_768, 4_096), 2_048);
        assert_eq!(bounded_prefill_step(2_048, 131_071, 4_096), 2_048);
        assert_eq!(bounded_prefill_step(2_048, 524_287, 4_096), 512);
        assert_eq!(bounded_prefill_step(2_048, 32_767, 2), 1);
        assert_eq!(bounded_prefill_step(2_048, usize::MAX, 4_096), 1);
    }
}
