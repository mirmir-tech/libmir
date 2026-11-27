use std::fs;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use uuid::Uuid;

use super::{AutoBf16Plan, candidate::candidates};
use crate::{
    CudaBackend, CudaConfig, CudaTuningConfig, CudaTuningMode, DenseExecution, DensePlanRequest,
    DenseRole, Error, ExecutionPhase, PlanSource, Result,
};

/// Decode routers measure split-K beside the other kernels and keep a
/// winner the candidate list offered, with correct results either way.
#[test]
fn router_decode_tunes_across_split_k_candidates() -> Result<()> {
    let directory =
        std::env::temp_dir().join(format!("libmir-cuda-dense-tuning-{}", Uuid::new_v4()));
    let backend = CudaBackend::new(CudaConfig {
        tuning: CudaTuningConfig {
            mode: CudaTuningMode::Startup,
            cache_directory: Some(directory.clone()),
            ..CudaTuningConfig::default()
        },
        ..CudaConfig::default()
    })?;
    let multiprocessors = backend.execution_planner().hardware().multiprocessor_count().get();
    for (output_features, input_features) in [(32, 2_048), (128, 4_096), (256, 7_168)] {
        let request = DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role: DenseRole::Router,
            tokens: 1,
            input_features,
            output_features,
        };
        let offered = candidates(request, multiprocessors);
        assert!(
            offered
                .iter()
                .any(|execution| matches!(execution, DenseExecution::SplitKVector { .. })),
            "{request:?} offers no split-K candidate: {offered:?}"
        );
        let input = copy(&backend, &vec![bf16::ONE; input_features])?;
        let weight = copy(&backend, &vec![bf16::ONE; input_features * output_features])?;
        let mut output = backend.pool().allocate_zeroed(backend.stream(), output_features)?;
        AutoBf16Plan::new(&backend, request)?.execute(&input, &weight, &mut output)?;
        let expected = bf16::from_f32(f32::from(u16::try_from(input_features)?));
        assert!(read(&backend, &output)?.iter().all(|value| *value == expected));
        let (selected, source) = backend
            .inner
            .tuner
            .lookup_dense(request)
            .ok_or(Error::InvalidExecutionPlan("router decode was not tuned"))?;
        assert!(offered.contains(&selected), "{selected:?} was not offered");
        assert_eq!(source, PlanSource::MeasuredStartup);
    }
    fs::remove_dir_all(directory)?;
    Ok(())
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read(backend: &CudaBackend, values: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    backend.stream().copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
