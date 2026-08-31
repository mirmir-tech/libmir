use models::semantic::{FeedForwardSpec, MixerSpec, SemanticModelSpec};

const DEFAULT_PREFILL_STEP: usize = 512;
const LARGE_PREFILL_STEP: usize = 2_048;

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
}
