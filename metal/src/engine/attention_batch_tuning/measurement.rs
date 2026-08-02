use std::time::{Duration, Instant};

use super::{Array, BatchAttentionExecution, KvContext, Result, Stream, execute_measured};

pub(super) fn measure(
    execution: BatchAttentionExecution,
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    causal: bool,
    stream: &Stream,
) -> Result<Duration> {
    let config = &stream.config().tuning;
    for _ in 0..config.warmup_iterations {
        let output = execute_measured(execution, queries, contexts, scale, causal, stream, true)?;
        evaluate(&output, stream)?;
    }
    let iterations = config.measurement_iterations.max(1);
    let started = Instant::now();
    for _ in 0..iterations {
        let output = execute_measured(execution, queries, contexts, scale, causal, stream, true)?;
        evaluate(&output, stream)?;
    }
    Ok(started.elapsed() / iterations)
}

fn evaluate(output: &[Array], stream: &Stream) -> Result<()> {
    let refs = output.iter().collect::<Vec<_>>();
    Array::concatenate(&refs, 0, stream)?.async_eval()?;
    stream.synchronize()
}
