use std::time::{Duration, Instant};

use super::{Array, Result, RoutingExecution, RoutingPaths, Stream, execute};

const MINIMUM_BALANCED_ITERATIONS: u32 = 5;

pub(super) fn measure_candidates(
    executions: [RoutingExecution; 5],
    routes: [&Array; 3],
    stream: &Stream,
    paths: RoutingPaths<'_>,
) -> Result<Vec<Vec<Duration>>> {
    let config = &stream.config().tuning;
    for indices in routes {
        for execution in &executions {
            for _ in 0..config.warmup_iterations {
                measure_once(*execution, indices, stream, paths)?;
            }
        }
    }
    let iterations = config.measurement_iterations.max(MINIMUM_BALANCED_ITERATIONS);
    let mut samples = vec![vec![Vec::with_capacity(iterations as usize); routes.len()]; 5];
    for iteration in 0..iterations as usize {
        for (scenario, indices) in routes.into_iter().enumerate() {
            let start = (iteration + scenario) % executions.len();
            for offset in 0..executions.len() {
                let candidate = (start + offset) % executions.len();
                samples[candidate][scenario].push(measure_once(
                    executions[candidate],
                    indices,
                    stream,
                    paths,
                )?);
            }
        }
    }
    Ok(samples
        .into_iter()
        .map(|scenarios| scenarios.into_iter().map(median).collect())
        .collect())
}

fn measure_once(
    execution: RoutingExecution,
    indices: &Array,
    stream: &Stream,
    paths: RoutingPaths<'_>,
) -> Result<Duration> {
    let started = Instant::now();
    execute(execution, indices, paths)?.async_eval(stream)?;
    stream.synchronize()?;
    Ok(started.elapsed())
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}
