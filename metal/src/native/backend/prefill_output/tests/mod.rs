mod abandonment;
mod cases;
mod recovery;

use std::time::{Duration, Instant};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::{
    backend::{ModelHandle, PrefillOutput, PrefillRequest, SamplingLogits},
    kv::BlockTable,
    tuning::TuningMode,
};
use uuid::Uuid;

use crate::native::backend::MetalBackend;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn manifest(path: String) -> ModelManifest {
    ModelManifest {
        id: format!("prefill-timing-{}", Uuid::new_v4()),
        path,
        tokenizer_path: None,
        context_len: 256,
        preferred_backends: vec![BackendTarget::Metal],
        quantization: Quantization::Int4,
    }
}

fn request(model: &ModelHandle, seed: u32) -> PrefillRequest {
    PrefillRequest {
        model: model.clone(),
        session_id: Uuid::new_v4(),
        prompt_tokens: (0..64).map(|i| (i + seed) % 64).collect(),
        cache_checkpoints: Vec::new(),
        block_table: BlockTable::with_block_size(16),
        cached_tokens: 0,
        generation_tokens: None,
        sampling_logits: SamplingLogits::None,
    }
}

fn check(output: &PrefillOutput, outer_elapsed: Duration) -> Result<u32> {
    let timings = output.timings.ok_or("Metal dropped prefill timings")?;
    assert!(timings.backend_execution > Duration::ZERO);
    assert!(timings.backend_execution + timings.backend_wait <= outer_elapsed);
    assert_eq!(timings.scheduler_queue, Duration::ZERO);
    assert_eq!(timings.cache_prepare, Duration::ZERO);
    assert_eq!(output.accepted_tokens, 64);
    Ok(output.next_token.ok_or("no greedy token")?)
}

fn exercise(manifest: &ModelManifest) -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.mode = TuningMode::Disabled;
    config.cache.prefix_cache_entries = 4;
    config.set_max_batch_requests(2);
    let backend = MetalBackend::new(config);
    let model = backend.load_model_with_progress(manifest, &mut |_| {})?;
    let mut reference = Vec::new();
    // Scalar miss and exact hit, through the real worker/output adapter.
    for seed in [0, 3] {
        for hit in [false, true] {
            let request = request(&model, seed);
            let started = Instant::now();
            let mut initial = None;
            let output = backend.prefill_request_with_progress(&request, &mut |event| {
                if let crate::MetalProgressEvent::PrefillTokens { count, .. } = event {
                    initial.get_or_insert_with(|| count.current());
                }
            })?;
            let token = check(&output, started.elapsed())?;
            assert_eq!(
                initial,
                Some(if hit {
                    64
                } else {
                    0
                })
            );
            if hit {
                assert_eq!(Some(&token), reference.last());
            } else {
                reference.push(token);
            }
            backend.release_session(&model, request.session_id)?;
        }
    }
    backend.clear_prefix_cache(&model)?;
    // Packed cold prefill followed by cohort-leased exact hits.
    for hit in [false, true] {
        let requests = [request(&model, 0), request(&model, 3)];
        let cohort = hit.then(|| backend.prepare_prefill_cohort(&requests)).transpose()?;
        let started = Instant::now();
        let mut initial = [None; 2];
        let batch =
            backend.prepare_prefill_batch(&requests, cohort.as_ref(), &mut |row, event| {
                if let crate::MetalProgressEvent::PrefillTokens { count, .. } = event {
                    initial[row].get_or_insert_with(|| count.current());
                }
            })?;
        assert_eq!(
            initial,
            [Some(if hit {
                64
            } else {
                0
            }); 2]
        );
        let mut complete = false;
        for _ in 0..128 {
            if backend
                .execute_generation_step(None, Some(&batch), 16, &mut |_, _| {})?
                .prefill?
            {
                complete = true;
                break;
            }
        }
        assert!(complete, "prefill batch made no progress");
        let outputs = backend.finish_prefill_batch(batch)?;
        let elapsed = started.elapsed();
        assert_eq!(outputs.len(), 2);
        for ((request, output), expected) in requests.iter().zip(&outputs).zip(&reference) {
            assert_eq!(check(output, elapsed)?, *expected);
            backend.release_session(&model, request.session_id)?;
        }
    }
    assert!(backend.unload_model(&model)?);
    Ok(())
}
