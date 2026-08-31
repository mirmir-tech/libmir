use runtime::{
    backend::{DecodeOutput, SamplingLogits, TokenEvent},
    metrics::GenerationMetricsRecorder,
    sampling::{Sampler, SamplerConfig},
};

use super::*;

#[test]
fn injects_reasoning_exit_once_after_a_cycle() -> Result<()> {
    let settings = settings();
    let mut recovery = CycleRecovery::new(
        settings,
        Some(7),
        256,
        SamplingLogits::None,
        Some((192, vec![91, 92])),
    )?;
    let mut generated = (0..168).collect::<Vec<_>>();
    generated.extend([7, 11, 13, 17, 19, 23, 29, 31].repeat(3));
    recovery.observe(&[], &generated);

    let mut metrics = GenerationMetricsRecorder::new();
    let mut sampler = Sampler::new(SamplerConfig::default())?;
    let output = output(77);
    assert_eq!(recovery.choose(&mut metrics, &output, &[], &mut sampler)?, 91);
    assert_eq!(recovery.choose(&mut metrics, &output, &[], &mut sampler)?, 92);
    assert_eq!(recovery.choose(&mut metrics, &output, &[], &mut sampler)?, 77);
    let metrics = metrics.snapshot(runtime::kv::CacheStats::default());
    assert_eq!(metrics.tokens.reasoning_exits, 1);
    assert_eq!(metrics.tokens.reasoning_exit_tokens, 2);
    Ok(())
}

fn settings() -> GenerationSettings {
    GenerationSettings {
        max_tokens: 512,
        min_tokens: 0,
        ignore_eos: false,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    }
}

fn output(token: u32) -> DecodeOutput {
    DecodeOutput {
        event: TokenEvent {
            token_id: Some(token),
            text: String::new(),
            finished: false,
        },
        logits: None,
        candidates: None,
        timings: None,
    }
}
