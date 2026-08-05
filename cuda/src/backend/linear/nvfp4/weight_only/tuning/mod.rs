use mircuda::{Context, DeviceBuffer, MarlinNvFp4ThreadConfig, Stream, bf16};
use runtime::tuning::select_fastest_candidate;

use super::{
    CudaBackend, NvFp4Config, NvFp4WeightOnly, NvFp4WeightOnlyTensorCore, NvFp4WeightOnlyWeight,
    marlin::MarlinNvFp4Bf16Linear,
};

mod measure;
use crate::{
    PlanSource, Result,
    backend::tuning::{CudaAutoTuner, QuantizedProfileExecution, QuantizedProfileRequest},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Execution {
    Compressed,
    TensorCore,
    MarlinN128K128,
    MarlinN128K64,
    MarlinN64K128,
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
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        config: NvFp4Config,
        marlin_available: bool,
    ) -> Result<Self> {
        let request = QuantizedProfileRequest::nvfp4_bf16_weight_only(
            tokens,
            config.input_features,
            config.output_features,
        );
        let cached = backend.auto_tuner().lookup_quantized(request).and_then(|(value, source)| {
            let QuantizedProfileExecution::NvFp4WeightOnly(value) = value else {
                return None;
            };
            let execution: Execution = value.into();
            if execution.is_marlin() && !marlin_available {
                return None;
            }
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
            validation: backend.pool().allocate(
                backend.stream(),
                if tokens == 1 {
                    elements
                } else {
                    elements.min(12_288)
                },
            )?,
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
        mut marlin: Option<&mut MarlinNvFp4Bf16Linear>,
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
        let marlin_compatible = if let Some(marlin) = &mut marlin {
            super::validation::marlin_compatible(
                &self.context,
                stream,
                compressed,
                marlin,
                weight,
                input,
                output,
                &mut self.validation,
            )?
        } else {
            [false; 3]
        };
        if marlin.is_some() && !marlin_compatible.iter().all(|compatible| *compatible) {
            tracing::warn!(
                request = ?self.request,
                ?marlin_compatible,
                "rejected numerically incompatible CUDA NVFP4 W4A16 Marlin candidate"
            );
        }
        let result = self.measure_candidates(
            stream,
            compressed,
            tensor_core,
            tensor_core_compatible,
            marlin,
            marlin_compatible,
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
            .min_by_key(|(_, (_, duration))| *duration)
            .map_or(0, |(index, _)| index);
        let fallback = timings
            .iter()
            .position(|(execution, _)| *execution == self.fallback)
            .ok_or(crate::Error::InvalidExecutionPlan("NVFP4 fallback candidate is missing"))?;
        let durations = timings.iter().map(|(_, duration)| *duration).collect::<Vec<_>>();
        let selected = select_fastest_candidate(
            fastest,
            fallback,
            &durations,
            self.tuner.minimum_improvement_bps(),
        );
        let (execution, average) = timings[selected];
        self.selected = Some(execution);
        self.claimed = false;
        self.tuner.record_quantized(
            self.request,
            QuantizedProfileExecution::NvFp4WeightOnly(execution.into()),
            average,
            tuning_elapsed,
        );
        super::profile::trace(self.request, execution, PlanSource::MeasuredStartup, Some(average));
        Ok(())
    }
}

impl Execution {
    const fn is_marlin(self) -> bool {
        matches!(self, Self::MarlinN128K128 | Self::MarlinN128K64 | Self::MarlinN64K128)
    }
}

pub(super) const fn marlin_candidates() -> [(Execution, MarlinNvFp4ThreadConfig); 3] {
    [
        (Execution::MarlinN128K128, MarlinNvFp4ThreadConfig::N128K128),
        (Execution::MarlinN128K64, MarlinNvFp4ThreadConfig::N128K64),
        (Execution::MarlinN64K128, MarlinNvFp4ThreadConfig::N64K128),
    ]
}
