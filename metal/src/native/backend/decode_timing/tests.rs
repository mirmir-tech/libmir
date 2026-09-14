use std::time::{Duration, Instant};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::{
    backend::{DecodeOutput, DecodeSequence, PrefillRequest, SamplingLogits},
    kv::BlockTable,
    tuning::TuningMode,
};
use uuid::Uuid;

use super::super::{MetalBackend, batch::execute_loaded_decode, execution::execute_decode};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const QUEUED: Duration = Duration::from_millis(20);

#[test]
fn queue_and_execution_are_disjoint_wall_intervals() {
    let start = Instant::now();
    let timing = super::finish(start, start + QUEUED, start + QUEUED + Duration::from_millis(3));
    assert_eq!(timing.backend_wait, QUEUED);
    assert_eq!(timing.backend_execution, Duration::from_millis(3));
    assert_eq!(timing.device_execution, None);
}

fn check(output: &DecodeOutput, elapsed: Duration) -> Result<()> {
    let timing = output.timings.ok_or("missing enabled profiling")?;
    assert!(timing.backend_wait >= QUEUED, "queue was attributed to execution");
    assert!(timing.backend_execution > Duration::ZERO);
    assert!(timing.backend_execution <= elapsed, "execution includes synthetic queue delay");
    assert!(timing.backend_wait + timing.backend_execution <= elapsed + QUEUED);
    assert_eq!(timing.device_execution, None);
    Ok(())
}

fn exercise(path: String) -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.mode = TuningMode::Disabled;
    config.set_max_batch_requests(3);
    let backend = MetalBackend::new(config);
    let manifest = ModelManifest {
        id: format!("decode-timing-{}", Uuid::new_v4()),
        path,
        tokenizer_path: None,
        context_len: 256,
        preferred_backends: vec![BackendTarget::Metal],
        quantization: Quantization::Int4,
    };
    let model = backend.load_model_with_progress(&manifest, &mut |_| {})?;
    let mut sequences = Vec::new();
    for _ in 0..3 {
        let request = PrefillRequest {
            model: model.clone(),
            session_id: Uuid::new_v4(),
            prompt_tokens: (0..32).map(|i| i % 16).collect(),
            cache_checkpoints: Vec::new(),
            block_table: BlockTable::with_block_size(16),
            cached_tokens: 0,
            generation_tokens: None,
            sampling_logits: SamplingLogits::None,
        };
        let output = backend.prefill_request_with_progress(&request, &mut |_| {})?;
        sequences.push(DecodeSequence {
            session_id: request.session_id,
            token_id: output.next_token.ok_or("no token")?,
            block_table: request.block_table,
            sampling_logits: SamplingLogits::None,
        });
    }
    backend.with_model(&model.id, move |loaded| {
        let sequence = &mut sequences[0];
        let started = Instant::now();
        let output = execute_decode(
            loaded,
            "timing",
            sequence.session_id,
            sequence.token_id,
            &sequence.block_table,
            sequence.sampling_logits,
            sequence.sampling_logits,
            true,
            started.checked_sub(QUEUED).ok_or_else(|| {
                crate::native::error::Error::InvalidDecodeBatch(
                    "synthetic test clock underflow".into(),
                )
            })?,
        )?;
        // Convert external test errors at this test closure boundary.
        assert!(check(&output, started.elapsed()).is_ok());
        sequence.token_id =
            output.event.token_id.ok_or(crate::native::error::Error::NoPendingDecode)?;
        // Mixed output materialization must give all rows one batch duration.
        sequences[2].sampling_logits = SamplingLogits::Full;
        let started = Instant::now();
        let outputs = execute_loaded_decode(
            loaded,
            &sequences,
            true,
            true,
            started.checked_sub(QUEUED).ok_or_else(|| {
                crate::native::error::Error::InvalidDecodeBatch(
                    "synthetic test clock underflow".into(),
                )
            })?,
        )?;
        let elapsed = started.elapsed();
        for output in &outputs {
            assert!(check(output, elapsed).is_ok());
        }
        assert_eq!(
            outputs[0].timings.map(|t| t.backend_wait),
            outputs[2].timings.map(|t| t.backend_wait)
        );
        assert_eq!(
            outputs[0].timings.map(|t| t.backend_execution),
            outputs[2].timings.map(|t| t.backend_execution)
        );
        assert_eq!(outputs[0].timings.map(|t| t.batch_rows), Some(2));
        assert_eq!(outputs[2].timings.map(|t| t.batch_rows), Some(1));
        assert!(outputs[2].logits.is_some());
        Ok(())
    })?;
    backend.unload_model(&model)?;
    Ok(())
}

#[test]
fn separates_worker_wait_for_hybrid_scalar_and_mixed_decode() -> Result<()> {
    use crate::engine::hybrid_linear_moe::tests::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("hybrid-decode-timing-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"))?;
    let result = exercise(root.to_string_lossy().into_owned());
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
fn separates_worker_wait_for_clamped_scalar_and_mixed_decode() -> Result<()> {
    use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("clamped-decode-timing-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), true)?;
    let result = exercise(root.to_string_lossy().into_owned());
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
#[ignore = "actual checkpoint decode timing and mixed output adapters"]
fn reports_real_model_decode_timings() -> Result<()> {
    exercise(std::env::var("MIRMIR_BENCH_MODEL")?)
}
