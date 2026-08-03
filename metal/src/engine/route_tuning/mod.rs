use std::time::{Duration, Instant};

use runtime::tuning::{TuningMode, select_robust_candidate};

use super::{Array, Dtype, Error, Result, Stream};

#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
pub enum ExpertActivation {
    GeluApprox,
    Silu,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RoutingKey {
    pub route_bucket: usize,
    pub experts: usize,
    pub top_k: usize,
    pub input: usize,
    pub intermediate: usize,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
    pub activation: ExpertActivation,
    pub fused_unsorted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum RoutingExecution {
    Unsorted,
    SortedGraph,
    SortedFused,
    GroupedFused,
}

#[derive(Clone, Copy, Debug)]
pub struct RoutingSpec {
    pub experts: usize,
    pub intermediate: usize,
    pub group_size: i32,
    pub bits: i32,
    pub activation: ExpertActivation,
    pub fused_unsorted: bool,
}

pub(super) fn forward<G, F, K, U>(
    spec: RoutingSpec,
    input: &Array,
    indices: &Array,
    stream: &Stream,
    paths: (G, F, K, U),
) -> Result<Array>
where
    G: Fn(&Array) -> Result<Array>,
    F: Fn(&Array) -> Result<Array>,
    K: Fn(&Array) -> Result<Array>,
    U: Fn(&Array) -> Result<Array>,
{
    let (sorted_graph, sorted_fused, grouped_fused, unsorted) = paths;
    let key = key(spec, input, indices)?;
    let fallback = fallback(indices)?;
    let decision = {
        let tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if tuner.config().mode == TuningMode::Disabled {
            Some(fallback)
        } else if let Some(execution) = tuner.routing_decision(key) {
            Some(execution)
        } else if tuner.config().mode == TuningMode::Startup && tuner.routing_budget_available() {
            None
        } else {
            Some(fallback)
        }
    };
    let paths: [&dyn Fn(&Array) -> Result<Array>; 4] =
        [&sorted_graph, &sorted_fused, &grouped_fused, &unsorted];
    decision.map_or_else(
        || tune(key, fallback, indices, stream, paths),
        |execution| execute(execution, indices, paths),
    )
}

fn tune(
    key: RoutingKey,
    fallback: RoutingExecution,
    indices: &Array,
    stream: &Stream,
    paths: [&dyn Fn(&Array) -> Result<Array>; 4],
) -> Result<Array> {
    let started = Instant::now();
    let result = (|| {
        let executions = [
            RoutingExecution::SortedGraph,
            RoutingExecution::SortedFused,
            RoutingExecution::GroupedFused,
            RoutingExecution::Unsorted,
        ];
        let patterns = route_patterns(key, indices)?;
        let routes = [indices, &patterns.balanced, &patterns.hot_set];
        let timings = executions
            .map(|execution| {
                routes
                    .map(|indices| measure(execution, indices, stream, paths))
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let fallback_index =
            executions.iter().position(|execution| *execution == fallback).unwrap_or(0);
        let selected = select_robust_candidate(
            fallback_index,
            &timings,
            stream.config().tuning.minimum_improvement_bps,
        );
        let execution = executions[selected];
        record(key, execution, started.elapsed(), stream);
        tracing::info!(
            target: "libmir::metal::tuning",
            ?execution,
            route_bucket = key.route_bucket,
            experts = key.experts,
            top_k = key.top_k,
            input_features = key.input,
            intermediate_features = key.intermediate,
            timings_us = ?timings
                .iter()
                .map(|scenarios| scenarios
                    .iter()
                    .map(|duration| duration.as_secs_f64() * 1_000_000.0)
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            "selected Metal expert routing execution profile"
        );
        execute(execution, indices, paths)
    })();
    result.or_else(|error| {
        record(key, fallback, started.elapsed(), stream);
        tracing::warn!(
            target: "libmir::metal::tuning",
            %error,
            ?fallback,
            "Metal expert routing tuning failed; retaining shape fallback"
        );
        execute(fallback, indices, paths)
    })
}

fn measure(
    execution: RoutingExecution,
    indices: &Array,
    stream: &Stream,
    paths: [&dyn Fn(&Array) -> Result<Array>; 4],
) -> Result<Duration> {
    for _ in 0..stream.config().tuning.warmup_iterations {
        execute(execution, indices, paths)?.async_eval()?;
    }
    stream.synchronize()?;
    let iterations = stream.config().tuning.measurement_iterations.max(1);
    let started = Instant::now();
    for _ in 0..iterations {
        execute(execution, indices, paths)?.async_eval()?;
    }
    stream.synchronize()?;
    Ok(started.elapsed() / iterations)
}

fn execute(
    execution: RoutingExecution,
    indices: &Array,
    [sorted_graph, sorted_fused, grouped_fused, unsorted]: [&dyn Fn(&Array) -> Result<Array>; 4],
) -> Result<Array> {
    match execution {
        RoutingExecution::Unsorted => unsorted(indices),
        RoutingExecution::SortedGraph => sorted_graph(indices),
        RoutingExecution::SortedFused => sorted_fused(indices),
        RoutingExecution::GroupedFused => grouped_fused(indices),
    }
}

struct RoutePatterns {
    balanced: Array,
    hot_set: Array,
}

fn route_patterns(key: RoutingKey, indices: &Array) -> Result<RoutePatterns> {
    let shape = indices.shape()?;
    let assignments = elements(&shape)?;
    let balanced = (0..assignments)
        .map(|assignment| u32::try_from(assignment % key.experts).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    let hot_set = (0..assignments)
        .map(|assignment| u32::try_from(assignment % key.top_k).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    Ok(RoutePatterns {
        balanced: Array::from_u32(&balanced, &shape)?,
        hot_set: Array::from_u32(&hot_set, &shape)?,
    })
}

fn record(key: RoutingKey, execution: RoutingExecution, elapsed: Duration, stream: &Stream) {
    let mut tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    tuner.record_routing(key, execution, elapsed);
    tuner.persist();
}

fn key(spec: RoutingSpec, input: &Array, indices: &Array) -> Result<RoutingKey> {
    let input_shape = input.shape()?;
    let routing_shape = indices.shape()?;
    if input_shape.len() != 3 || routing_shape.len() != 3 || input_shape[..2] != routing_shape[..2]
    {
        return Err(Error::InvalidModel("expert input and routing shapes do not align".into()));
    }
    let input_width = usize::try_from(*input_shape.last().ok_or(Error::ShapeOverflow)?)?;
    let top_k = usize::try_from(*routing_shape.last().ok_or(Error::ShapeOverflow)?)?;
    let routes = elements(&routing_shape)?;
    Ok(RoutingKey {
        route_bucket: routes.checked_next_power_of_two().ok_or(Error::ShapeOverflow)?,
        experts: spec.experts,
        top_k,
        input: input_width,
        intermediate: spec.intermediate,
        group_size: spec.group_size,
        bits: spec.bits,
        dtype: input.dtype()?,
        activation: spec.activation,
        fused_unsorted: spec.fused_unsorted,
    })
}

fn fallback(indices: &Array) -> Result<RoutingExecution> {
    if elements(&indices.shape()?)? >= 64 {
        Ok(RoutingExecution::SortedGraph)
    } else {
        Ok(RoutingExecution::Unsorted)
    }
}

fn elements(shape: &[i32]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
    })
}

#[cfg(test)]
mod tests;
