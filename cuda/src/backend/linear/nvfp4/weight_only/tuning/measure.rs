use std::time::Duration;

use mircuda::{Context, DeviceBuffer, Stream, bf16};

use super::{Execution, Selection};
use crate::{Result, kernels::NvFp4WeightOnlyLaunch};

impl Selection {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn measure_candidates(
        &mut self,
        stream: &Stream,
        compressed: &super::NvFp4WeightOnly,
        tensor_core: &super::NvFp4WeightOnlyTensorCore,
        tensor_core_compatible: bool,
        marlin: Option<&mut super::MarlinNvFp4Bf16Linear>,
        marlin_compatible: [bool; 3],
        materialized: &mut super::super::super::super::Bf16Projection,
        weight: &super::NvFp4WeightOnlyWeight,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<(Vec<(Execution, Duration)>, Duration)> {
        let (warmup, iterations) = self.tuner.iterations(self.request.tokens());
        let compressed_time = measure(&self.context, stream, warmup, iterations, || {
            compressed.execute(stream, &mut launch(input, output, weight))
        })?;
        let mut timings = vec![(Execution::Compressed, compressed_time)];
        if tensor_core_compatible {
            timings.push((
                Execution::TensorCore,
                measure(&self.context, stream, warmup, iterations, || {
                    tensor_core.execute(stream, &mut launch(input, output, weight))
                })?,
            ));
        }
        if marlin_compatible.iter().any(|compatible| *compatible) {
            let marlin = marlin.ok_or(crate::Error::InvalidExecutionPlan(
                "validated dense Marlin candidate is unavailable",
            ))?;
            for ((execution, config), compatible) in
                super::marlin_candidates().into_iter().zip(marlin_compatible)
            {
                if !compatible {
                    continue;
                }
                timings.push((
                    execution,
                    measure(&self.context, stream, warmup, iterations, || {
                        marlin.execute(input, output, config)
                    })?,
                ));
            }
        }
        timings.push((
            Execution::Materialized,
            measure(&self.context, stream, warmup, iterations, || {
                materialized.execute(input, &weight.materialized, &mut self.scratch)
            })?,
        ));
        let executions = warmup.saturating_add(iterations);
        let tuning_elapsed = timings
            .iter()
            .map(|(_, duration)| *duration)
            .fold(Duration::ZERO, Duration::saturating_add)
            .saturating_mul(executions);
        Ok((timings, tuning_elapsed))
    }
}

fn launch<'a>(
    input: &'a DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
    weight: &'a super::NvFp4WeightOnlyWeight,
) -> NvFp4WeightOnlyLaunch<'a> {
    NvFp4WeightOnlyLaunch {
        input,
        weight: &weight.weight,
        block_scales: &weight.scales,
        global_scale: &weight.global_scale,
        output,
    }
}

fn measure(
    context: &Context,
    stream: &Stream,
    warmup: u32,
    iterations: u32,
    mut execute: impl FnMut() -> Result<()>,
) -> Result<Duration> {
    for _ in 0..warmup {
        execute()?;
    }
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..iterations {
        execute()?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(Duration::from_secs_f64(
        f64::from(started.elapsed_ms(&completed)?) / (f64::from(iterations) * 1_000.0),
    ))
}
