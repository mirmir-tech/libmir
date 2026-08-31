use models::generation::GenerationSettings;
use runtime::{
    backend::{DecodeOutput, SamplingLogits},
    metrics::GenerationMetricsRecorder,
    sampling::{Sampler, SamplerConfig},
};

use crate::Result;

mod detector;
#[cfg(test)]
mod tests;

use detector::CycleDetector;

const RECOVERY_PENALTY: f32 = 1.2;

pub(super) struct CycleRecovery {
    state: RecoveryState,
    reasoning_exit: Option<Vec<u32>>,
}

enum RecoveryState {
    Disabled,
    Watching {
        sampler: Sampler,
        detector: CycleDetector,
    },
    Active(ActiveRecovery),
    Exiting(ReasoningExit),
}

struct ActiveRecovery {
    sampler: Sampler,
    history: Vec<u32>,
    observed_tokens: usize,
    first: bool,
}

struct ReasoningExit {
    tokens: std::vec::IntoIter<u32>,
    first: bool,
}

impl CycleRecovery {
    pub(super) fn new(
        _settings: GenerationSettings,
        seed: Option<u64>,
        vocab_size: usize,
        sampling: SamplingLogits,
        reasoning_exit: Option<(usize, Vec<u32>)>,
    ) -> Result<Self> {
        let detector =
            reasoning_exit.as_ref().map_or_else(CycleDetector::default, |(min_tokens, _)| {
                CycleDetector::reasoning_exit(*min_tokens)
            });
        let state = if sampling == SamplingLogits::None {
            RecoveryState::Watching {
                sampler: recovery_sampler(seed, vocab_size)?,
                detector,
            }
        } else {
            RecoveryState::Disabled
        };
        Ok(Self {
            state,
            reasoning_exit: reasoning_exit.map(|(_, tokens)| tokens),
        })
    }

    pub(super) fn observe(&mut self, prompt: &[u32], generated: &[u32]) {
        if let RecoveryState::Active(recovery) = &mut self.state {
            if let Some(tokens) = generated.get(recovery.observed_tokens..) {
                recovery.history.extend_from_slice(tokens);
                recovery.observed_tokens = generated.len();
            }
            return;
        }
        let RecoveryState::Watching { detector, .. } = &mut self.state else {
            return;
        };
        let Some(detection) = detector.observe(generated) else {
            return;
        };
        let mut history = Vec::with_capacity(prompt.len() + generated.len());
        history.extend_from_slice(prompt);
        history.extend_from_slice(generated);
        let state = std::mem::replace(&mut self.state, RecoveryState::Disabled);
        let RecoveryState::Watching { sampler, .. } = state else {
            return;
        };
        if let Some(tokens) = self.reasoning_exit.take().filter(|tokens| !tokens.is_empty()) {
            tracing::warn!(
                generated_tokens = generated.len(),
                span = detection.span,
                kind = ?detection.kind,
                "reasoning cycle exit activated"
            );
            self.state =
                RecoveryState::Exiting(ReasoningExit { tokens: tokens.into_iter(), first: true });
            return;
        }
        tracing::warn!(
            generated_tokens = generated.len(),
            span = detection.span,
            kind = ?detection.kind,
            "generation cycle recovery activated"
        );
        self.state = RecoveryState::Active(ActiveRecovery {
            sampler,
            history,
            observed_tokens: generated.len(),
            first: true,
        });
    }

    pub(super) const fn sampling(&self, normal: SamplingLogits) -> SamplingLogits {
        match self.state {
            RecoveryState::Active(_) => SamplingLogits::Full,
            RecoveryState::Exiting(_) => SamplingLogits::None,
            RecoveryState::Disabled | RecoveryState::Watching { .. } => normal,
        }
    }

    pub(super) fn choose(
        &mut self,
        metrics: &mut GenerationMetricsRecorder,
        output: &DecodeOutput,
        normal_history: &[u32],
        normal_sampler: &mut Sampler,
    ) -> Result<u32> {
        match self.state {
            RecoveryState::Active(_) => self.choose_recovery(metrics, output),
            RecoveryState::Exiting(_) => self.choose_exit(metrics),
            RecoveryState::Disabled | RecoveryState::Watching { .. } => {
                super::sampling::choose_timed(
                    metrics,
                    output.event.token_id,
                    output.logits.as_ref(),
                    output.candidates.as_ref(),
                    normal_history,
                    normal_sampler,
                )
            },
        }
    }

    fn choose_recovery(
        &mut self,
        metrics: &mut GenerationMetricsRecorder,
        output: &DecodeOutput,
    ) -> Result<u32> {
        let state = std::mem::replace(&mut self.state, RecoveryState::Disabled);
        let RecoveryState::Active(mut recovery) = state else {
            return Err(runtime::RuntimeError::Backend(
                "cycle recovery lost its active state".into(),
            )
            .into());
        };
        if recovery.first {
            metrics.record_recovery_attempt();
            recovery.first = false;
        }
        let token = super::sampling::choose_timed(
            metrics,
            None,
            output.logits.as_ref(),
            output.candidates.as_ref(),
            &recovery.history,
            &mut recovery.sampler,
        )?;
        metrics.record_recovery_token();
        self.state = RecoveryState::Active(recovery);
        Ok(token)
    }

    fn choose_exit(&mut self, metrics: &mut GenerationMetricsRecorder) -> Result<u32> {
        let state = std::mem::replace(&mut self.state, RecoveryState::Disabled);
        let RecoveryState::Exiting(mut exit) = state else {
            return Err(
                runtime::RuntimeError::Backend("reasoning exit lost its state".into()).into()
            );
        };
        let token = exit.tokens.next().ok_or_else(|| {
            runtime::RuntimeError::Backend("reasoning exit token sequence is empty".into())
        })?;
        if exit.first {
            metrics.record_reasoning_exit();
            exit.first = false;
        }
        metrics.record_reasoning_exit_token();
        if !exit.tokens.as_slice().is_empty() {
            self.state = RecoveryState::Exiting(exit);
        }
        Ok(token)
    }
}

fn recovery_sampler(seed: Option<u64>, vocab_size: usize) -> Result<Sampler> {
    Ok(Sampler::new(SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: RECOVERY_PENALTY,
        vocab_size: Some(vocab_size),
        seed: seed.unwrap_or_else(|| SamplerConfig::default().seed),
    })?)
}

#[cfg(test)]
pub(super) fn repeated_cycle(tokens: &[u32]) -> Option<usize> {
    CycleDetector::default().observe(tokens).map(|detection| detection.span)
}
