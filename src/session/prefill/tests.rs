#![allow(clippy::print_stderr, reason = "ignored real-model regression emits artifact evidence")]

use std::{
    sync::{Arc, Barrier},
    time::Instant,
};

use runtime::{backend::SamplingLogits, tuning::TuningMode};

use crate::{CancellationToken, GenerationOverrides, Library, ProgressEvent, RuntimeConfig};

#[test]
#[ignore = "real-model public API cancellation during concurrent prefill, followed by decode and refill"]
fn cancels_real_prefill_without_losing_a_sibling() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("MIRMIR_BENCH_MODEL")?;
    let mut config = RuntimeConfig {
        automatic_kv_cache: false,
        ..RuntimeConfig::default()
    };
    config.kv_cache.block_count = 4096;
    config.scheduler.max_batch_requests = 2;
    config.scheduler.max_batch_tokens = 2048;
    config.scheduler.prefill_batch_wait_us = 1000;
    config.metal.tuning.mode = TuningMode::Disabled;
    config.metal.cache.prefix_cache_entries = 0;
    let model = Library::new(config).load(path, GenerationOverrides::default(), &mut |_| {})?;
    let barrier = Arc::new(Barrier::new(2));
    let cancelled = CancellationToken::new();
    let prompt_len = 8193;
    // Deliberately different prefixes: both requests must enter execution rather
    // than one waiting for the other's cache-fill leadership.
    std::thread::scope(|scope| -> crate::Result<()> {
        let other = model.clone();
        let ready = barrier.clone();
        let sibling = scope.spawn(move || -> crate::Result<()> {
            let tokens = (0..prompt_len)
                .map(|i| u32::try_from(i % 64 + 80).unwrap_or_default())
                .collect::<Vec<_>>();
            let mut session = other.session();
            ready.wait();
            let output = session.prefill(&tokens, SamplingLogits::None, &mut |_| {})?;
            assert_eq!(output.accepted_tokens, prompt_len);
            let token = output.next_token.ok_or_else(missing_token)?;
            let decoded = session.decode(token, SamplingLogits::None)?;
            assert!(decoded.event.token_id.is_some());
            Ok(())
        });
        let tokens = (0..prompt_len)
            .map(|i| u32::try_from(i % 64 + 16).unwrap_or_default())
            .collect::<Vec<_>>();
        let mut session = model.session();
        let mut stopped_at = None;
        let mut last_progress = 0;
        barrier.wait();
        let result = session.prefill_cancellable(
            &tokens,
            SamplingLogits::None,
            &mut |event| {
                if let ProgressEvent::PrefillTokens { count, .. } = event {
                    last_progress = count.current();
                    if count.current() > 0 && stopped_at.is_none() {
                        stopped_at = Some((Instant::now(), count.current()));
                        cancelled.cancel();
                    }
                }
            },
            &cancelled,
        );
        assert!(
            matches!(result, Err(crate::Error::Cancelled)),
            "unexpected cancellation outcome: {result:?}"
        );
        let (requested, processed) = stopped_at.ok_or_else(missing_token)?;
        let latency = requested.elapsed();
        assert!(last_progress < prompt_len as u64, "cancelled row processed its complete prompt");
        assert!(
            last_progress <= processed + 2048,
            "more than one in-flight scheduler quantum ran after cancellation"
        );
        drop(session);
        eprintln!(
            "prefill_cancel requested_at={processed} last_progress={last_progress} total={prompt_len} acknowledgement_ms={:.3}",
            latency.as_secs_f64() * 1000.0
        );
        sibling.join().map_err(|_| missing_token())??;
        Ok(())
    })?;
    let mut refill = model.session();
    let output = refill.prefill(&[23; 65], SamplingLogits::None, &mut |_| {})?;
    let token = output.next_token.ok_or_else(missing_token)?;
    assert!(refill.decode(token, SamplingLogits::None)?.event.token_id.is_some());
    drop(refill);
    eprintln!(
        "prefill_cancel sibling_prefill_and_decode=ok refill_and_decode=ok cache={:?}",
        model.cache_stats()
    );
    assert_eq!(
        model.cache_stats().used_blocks,
        model.cache_stats().cached_prefixes,
        "released sessions still own uncached logical blocks"
    );
    cancel_generation(&model)?;
    model.unload()?;
    Ok(())
}

fn missing_token() -> crate::Error {
    runtime::RuntimeError::Backend(
        "prefill cancellation fixture did not complete its expected stage".into(),
    )
    .into()
}

fn cancel_generation(model: &crate::Model) -> crate::Result<()> {
    let cancellation = CancellationToken::new();
    let request = crate::GenerationRequest {
        conversation: crate::Conversation {
            messages: vec![crate::Message {
                role: "user".into(),
                content: "Explain this log. ".repeat(3000),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: crate::ToolChoice::default(),
        },
        options: GenerationOverrides {
            max_tokens: Some(16),
            temperature: Some(0.0),
            ..GenerationOverrides::default()
        },
        ..crate::GenerationRequest::default()
    };
    let mut cancelled_at = None;
    let mut last_count = runtime::progress::ProgressCount::new(0, 0);
    let mut published = false;
    let generated = model.generate_cancellable(
        &request,
        &mut |event| {
            if let ProgressEvent::PrefillTokens { count, .. } = event {
                last_count = count;
                if count.current() > 0 && cancelled_at.is_none() {
                    cancelled_at = Some(Instant::now());
                    cancellation.cancel();
                }
            }
        },
        &mut |_| published = true,
        &cancellation,
    );
    assert!(matches!(generated, Err(crate::Error::Cancelled)));
    assert!(!published, "cancelled generation published a token");
    assert!(
        last_count.current() < last_count.total(),
        "generation ignored cancellation until the prompt ended"
    );
    eprintln!(
        "generation_prefill_cancel last_progress={} total={} acknowledgement_ms={:.3}",
        last_count.current(),
        last_count.total(),
        cancelled_at.ok_or_else(missing_token)?.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
