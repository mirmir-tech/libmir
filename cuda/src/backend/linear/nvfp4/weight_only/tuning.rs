use std::time::Duration;

use mircuda::{Context, DeviceBuffer, Stream, bf16};
use runtime::tuning::select_fastest_candidate;

use super::{
    CudaBackend, NvFp4Config, NvFp4WeightOnly, NvFp4WeightOnlyTensorCore, NvFp4WeightOnlyWeight,
};
use crate::{
    PlanSource, Result,
    backend::tuning::{CudaAutoTuner, QuantizedProfileExecution, QuantizedProfileRequest},
    kernels::NvFp4WeightOnlyLaunch,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Execution {
    Compressed,
    TensorCore,
    Materialized,
}

#[derive(Debug)]
pub(super) struct Selection {
    request: QuantizedProfileRequest,
    selected: Option<Execution>,
    fallback: Execution,
    claimed: bool,
    scratch: DeviceBuffer<bf16>,
    validation: DeviceBuffer<bf16>,
    context: Context,
    tuner: CudaAutoTuner,
}

impl Selection {
    pub(super) fn new(backend: &CudaBackend, tokens: usize, config: NvFp4Config) -> Result<Self> {
        let request = QuantizedProfileRequest::nvfp4_bf16_weight_only(
            tokens,
            config.input_features,
            config.output_features,
        );
        let cached = backend.auto_tuner().lookup_quantized(request).and_then(|(value, source)| {
            let QuantizedProfileExecution::NvFp4WeightOnly(value) = value else {
                return None;
            };
            let execution = value.into();
            super::profile::trace(request, execution, source, None);
            Some(execution)
        });
        let claimed = cached.is_none() && backend.auto_tuner().claim_quantized(request);
        let elements = tokens.checked_mul(config.output_features).ok_or(
            crate::Error::InvalidExecutionPlan("NVFP4 W4A16 tuning scratch size overflows"),
        )?;
        Ok(Self {
            request,
            selected: cached,
            fallback: if tokens == 1 {
                Execution::Compressed
            } else {
                Execution::Materialized
            },
            claimed,
            scratch: backend.pool().allocate(backend.stream(), elements)?,
            validation: backend.pool().allocate(backend.stream(), elements.min(4_096))?,
            context: backend.context().clone(),
            tuner: backend.auto_tuner().clone(),
        })
    }

    pub(super) fn execution(&self) -> Execution {
        self.selected.unwrap_or(self.fallback)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn select(
        &mut self,
        stream: &Stream,
        compressed: &NvFp4WeightOnly,
        tensor_core: &NvFp4WeightOnlyTensorCore,
        materialized: &mut super::super::super::Bf16Projection,
        weight: &NvFp4WeightOnlyWeight,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.selected.is_some() || !self.claimed {
            return Ok(());
        }
        let tensor_core_compatible = super::validation::tensor_core_compatible(
            &self.context,
            stream,
            compressed,
            tensor_core,
            weight,
            input,
            output,
            &mut self.validation,
        )?;
        if !tensor_core_compatible {
            tracing::warn!(
                request = ?self.request,
                "rejected numerically incompatible CUDA NVFP4 W4A16 Tensor Core candidate"
            );
        }
        let result = self.measure_candidates(
            stream,
            compressed,
            tensor_core,
            tensor_core_compatible,
            materialized,
            weight,
            input,
            output,
        );
        let (timings, tuning_elapsed) = match result {
            Ok(value) => value,
            Err(error) => {
                self.tuner.abandon_quantized(self.request);
                self.claimed = false;
                self.selected = Some(self.fallback);
                tracing::warn!(
                    request = ?self.request,
                    %error,
                    "CUDA NVFP4 W4A16 tuning failed; retaining the stable fallback"
                );
                return Ok(());
            },
        };
        let fastest = timings
            .iter()
            .enumerate()
            .min_by_key(|(_, duration)| **duration)
            .map_or(0, |(index, _)| index);
        let fallback = match self.fallback {
            Execution::Compressed => 0,
            Execution::TensorCore => 1,
            Execution::Materialized => 2,
        };
        let selected = select_fastest_candidate(
            fastest,
            fallback,
            &timings,
            self.tuner.minimum_improvement_bps(),
        );
        let execution =
            [Execution::Compressed, Execution::TensorCore, Execution::Materialized][selected];
        self.selected = Some(execution);
        self.claimed = false;
        self.tuner.record_quantized(
            self.request,
            QuantizedProfileExecution::NvFp4WeightOnly(execution.into()),
            timings[selected],
            tuning_elapsed,
        );
        super::profile::trace(
            self.request,
            execution,
            PlanSource::MeasuredStartup,
            Some(timings[selected]),
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn measure_candidates(
        &mut self,
        stream: &Stream,
        compressed: &NvFp4WeightOnly,
        tensor_core: &NvFp4WeightOnlyTensorCore,
        tensor_core_compatible: bool,
        materialized: &mut super::super::super::Bf16Projection,
        weight: &NvFp4WeightOnlyWeight,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<([Duration; 3], Duration)> {
        let (warmup, iterations) = self.tuner.iterations(self.request_tokens());
        let compressed_time = measure(&self.context, stream, warmup, iterations, || {
            compressed.execute(
                stream,
                &mut NvFp4WeightOnlyLaunch {
                    input,
                    weight: &weight.weight,
                    block_scales: &weight.scales,
                    global_scale: &weight.global_scale,
                    output,
                },
            )
        })?;
        let tensor_core_time = if tensor_core_compatible {
            measure(&self.context, stream, warmup, iterations, || {
                tensor_core.execute(
                    stream,
                    &mut NvFp4WeightOnlyLaunch {
                        input,
                        weight: &weight.weight,
                        block_scales: &weight.scales,
                        global_scale: &weight.global_scale,
                        output,
                    },
                )
            })?
        } else {
            Duration::MAX
        };
        let materialized_time = measure(&self.context, stream, warmup, iterations, || {
            materialized.execute(input, &weight.materialized, &mut self.scratch)
        })?;
        let executions = u32::from(warmup).saturating_add(iterations);
        let timings = [compressed_time, tensor_core_time, materialized_time];
        let tuning_elapsed = timings
            .iter()
            .copied()
            .filter(|duration| *duration != Duration::MAX)
            .fold(Duration::ZERO, Duration::saturating_add)
            .saturating_mul(executions);
        Ok((timings, tuning_elapsed))
    }

    const fn request_tokens(&self) -> usize {
        self.request.tokens()
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
    Ok(Duration::from_secs_f32(
        started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
    ))
}
