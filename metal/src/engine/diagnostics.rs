use std::time::Instant;

use super::memory_stats;

#[derive(Debug, Clone, Copy)]
pub enum Stage {
    Forward,
    Segment,
    Evaluate,
    Synchronize,
    DetachArenas,
    DetachState,
    Reclaim,
    Checkpoint,
    Snapshot,
    FirstToken,
}

/// Observe existing boundaries only. Evaluation includes MLX scheduling and GPU
/// completion; it is not a GPU-only kernel timer. Forward includes nested
/// Segment events; subtract those to exclude known in-forward execution.
/// Allocator reads do not sync.
pub fn measure<T, E>(stage: Stage, operation: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    if !tracing::enabled!(target: "libmir::metal::prefill", tracing::Level::DEBUG) {
        return operation();
    }
    let before = memory_stats().ok();
    tracing::debug!(target: "libmir::metal::prefill", ?stage, ?before, "prefill stage started");
    let started = Instant::now();
    let result = operation();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let after = memory_stats().ok();
    tracing::debug!(target: "libmir::metal::prefill", ?stage, elapsed_ms,
        success = result.is_ok(), ?after, "prefill stage finished");
    result
}

/// Preserve the packed prefill command boundary used by routed decoders.
pub fn evaluate_segment(hidden: &super::Array, stream: &super::Stream) -> super::Result<()> {
    measure(Stage::Segment, || {
        hidden.async_eval(stream)?;
        stream.synchronize()
    })
}

#[cfg(test)]
pub fn gdn_experiment_calls() -> usize {
    super::kernels::gated_delta::experiment::calls()
}
