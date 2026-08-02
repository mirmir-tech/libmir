use std::time::Duration;

use runtime::tuning::materially_faster;

use crate::AttentionExecution;

pub struct SplitMeasurement {
    pub partition_tokens: usize,
    pub timings: Vec<Duration>,
    pub score: Duration,
}

pub fn select_execution(
    fallback: AttentionExecution,
    fallback_partition: usize,
    max_context_tokens: usize,
    contexts: &[usize],
    direct: &[Duration],
    splits: &[SplitMeasurement],
    minimum_bps: u16,
) -> AttentionExecution {
    let Some(fastest) = splits.iter().min_by_key(|candidate| candidate.score) else {
        return fallback;
    };
    let baseline = splits
        .iter()
        .find(|candidate| candidate.partition_tokens == fallback_partition)
        .unwrap_or(fastest);
    let selected = if materially_faster(fastest.score, baseline.score, minimum_bps) {
        fastest
    } else {
        baseline
    };
    let crossover = contexts.iter().copied().zip(direct).zip(&selected.timings).find_map(
        |((tokens, direct), split)| {
            materially_faster(*split, *direct, minimum_bps).then_some(tokens)
        },
    );
    match fallback {
        AttentionExecution::Direct => {
            crossover.map_or(AttentionExecution::Direct, |threshold| AttentionExecution::SplitKv {
                partition_tokens: selected.partition_tokens,
                threshold_tokens: threshold,
            })
        },
        AttentionExecution::SplitKv { threshold_tokens, .. } => {
            let observed_limit = contexts.last().copied().unwrap_or(0);
            let threshold_tokens = crossover.unwrap_or_else(|| {
                observed_limit
                    .saturating_add(1)
                    .min(max_context_tokens.saturating_add(1))
                    .max(threshold_tokens)
            });
            AttentionExecution::SplitKv {
                partition_tokens: selected.partition_tokens,
                threshold_tokens,
            }
        },
    }
}

pub fn execution_average(
    execution: AttentionExecution,
    direct: &[Duration],
    splits: &[SplitMeasurement],
) -> Duration {
    let timings = match execution {
        AttentionExecution::Direct => direct,
        AttentionExecution::SplitKv { partition_tokens, .. } => splits
            .iter()
            .find(|candidate| candidate.partition_tokens == partition_tokens)
            .map_or(direct, |candidate| candidate.timings.as_slice()),
    };
    let count = u32::try_from(timings.len()).unwrap_or(u32::MAX).max(1);
    timings.iter().copied().sum::<Duration>() / count
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{SplitMeasurement, select_execution};
    use crate::AttentionExecution;

    fn split(partition_tokens: usize, micros: &[u64]) -> SplitMeasurement {
        let timings = micros.iter().map(|value| Duration::from_micros(*value)).collect::<Vec<_>>();
        let score = timings.iter().copied().sum();
        SplitMeasurement { partition_tokens, timings, score }
    }

    #[test]
    fn learns_partition_and_first_material_crossover() {
        let selected = select_execution(
            AttentionExecution::SplitKv {
                partition_tokens: 256,
                threshold_tokens: 65,
            },
            256,
            4_096,
            &[64, 256, 1_024],
            &[Duration::from_micros(10), Duration::from_micros(30), Duration::from_micros(100)],
            &[split(128, &[20, 20, 40]), split(256, &[25, 28, 70])],
            300,
        );
        assert_eq!(
            selected,
            AttentionExecution::SplitKv {
                partition_tokens: 128,
                threshold_tokens: 256
            }
        );
    }

    #[test]
    fn direct_fallback_is_retained_without_a_crossover() {
        let selected = select_execution(
            AttentionExecution::Direct,
            256,
            512,
            &[64, 256],
            &[Duration::from_micros(10), Duration::from_micros(20)],
            &[split(256, &[20, 30])],
            300,
        );
        assert_eq!(selected, AttentionExecution::Direct);
    }
}
