use runtime::kv::BlockTable;
use uuid::Uuid;

use super::super::CudaMoeModelSession;
use crate::{CudaBackend, Result};

#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
pub(super) fn run(
    backend: &CudaBackend,
    sequential: &mut CudaMoeModelSession,
    prefill: &mut CudaMoeModelSession,
    prompt: &[u32],
    table: &mut BlockTable,
) -> Result<()> {
    const REPEATS: usize = 10;
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    let decode_profile = std::env::var_os("LIBMIR_CUDA_PROFILE_DECODE").is_some();
    let profiler = decode_profile
        .then(|| backend.inner.context.start_profiler_range())
        .transpose()?;
    started.record(&backend.inner.stream)?;
    for _ in 0..REPEATS {
        for (index, token) in prompt.iter().copied().enumerate() {
            table.set_token_len(index + 1);
            sequential.decode(Uuid::nil(), token, table)?;
        }
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    if let Some(profiler) = profiler {
        profiler.stop()?;
    }
    let sequential_ms = started.elapsed_ms(&completed)? / REPEATS as f32;

    table.set_token_len(prompt.len());
    prefill.prefill_from(Uuid::nil(), prompt, 0, table)?;
    backend.inner.stream.synchronize()?;
    let profiler = (!decode_profile)
        .then(|| backend.inner.context.start_profiler_range())
        .transpose()?;
    started.record(&backend.inner.stream)?;
    for _ in 0..REPEATS {
        prefill.prefill_from(Uuid::nil(), prompt, 0, table)?;
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    if let Some(profiler) = profiler {
        profiler.stop()?;
    }
    let prefill_ms = started.elapsed_ms(&completed)? / REPEATS as f32;
    let tokens = prompt.len() as f32;
    eprintln!(
        "sequential prefill: {sequential_ms:.3} ms, {:.1} tok/s",
        tokens * 1_000.0 / sequential_ms
    );
    eprintln!(
        "batched prefill:    {prefill_ms:.3} ms, {:.1} tok/s",
        tokens * 1_000.0 / prefill_ms
    );
    Ok(())
}
