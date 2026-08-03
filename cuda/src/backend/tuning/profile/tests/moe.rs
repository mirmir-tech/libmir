use super::*;

#[test]
fn moe_profile_survives_with_routing_geometry() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = MoeProfileRequest::nvfp4(
        MoePlanRequest::nvfp4(ExecutionPhase::Decode, 1, 128, 8, 2_048, 768),
        GatedActivation::GeluTanh,
        false,
    );
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_moe(request));
    tuner.record_moe(
        request,
        MoeProfileExecution::NvFp4(MoeExecution::IndexedGrouped),
        Duration::from_micros(91),
        Duration::from_millis(3),
    );

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(
        cached.lookup_moe(request),
        Some((
            MoeProfileExecution::NvFp4(MoeExecution::IndexedGrouped),
            PlanSource::MeasuredCache
        ))
    );
    let payload = profile_payload(&directory)?;
    assert!(payload.contains("\"moe\""));
    assert!(payload.contains("\"experts\": 128"));
    assert!(payload.contains("\"top_k\": 8"));
    let weight_only = MoeProfileRequest::nvfp4(
        MoePlanRequest::nvfp4(ExecutionPhase::Decode, 1, 128, 8, 2_048, 768),
        GatedActivation::GeluTanh,
        true,
    );
    assert_eq!(cached.lookup_moe(weight_only), None);
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn affine_moe_profile_isolated_by_storage_and_activation() -> Result<(), Box<dyn std::error::Error>>
{
    let directory = temporary_directory();
    let request = MoeProfileRequest::affine(
        ExecutionPhase::Decode,
        1,
        256,
        8,
        2_048,
        768,
        64,
        4,
        GatedActivation::Silu,
    );
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_moe(request));
    tuner.record_moe(
        request,
        MoeProfileExecution::Affine(AffineMoeExecution::SeparatePair),
        Duration::from_micros(73),
        Duration::from_millis(2),
    );

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(
        cached.lookup_moe(request),
        Some((
            MoeProfileExecution::Affine(AffineMoeExecution::SeparatePair),
            PlanSource::MeasuredCache,
        ))
    );
    let different = MoeProfileRequest::affine(
        ExecutionPhase::Decode,
        1,
        256,
        8,
        2_048,
        768,
        32,
        4,
        GatedActivation::Silu,
    );
    assert_eq!(cached.lookup_moe(different), None);
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn clamped_moe_profile_isolated_by_physical_storage() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = MoeProfileRequest::clamped(
        ExecutionPhase::Prefill,
        128,
        32,
        4,
        2_880,
        2_880,
        ClampedMoeStorage::Native,
    );
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_moe(request));
    tuner.record_moe(
        request,
        MoeProfileExecution::Clamped(ClampedMoeExecution::RouteParallel),
        Duration::from_micros(66),
        Duration::from_millis(4),
    );

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(
        cached.lookup_moe(request),
        Some((
            MoeProfileExecution::Clamped(ClampedMoeExecution::RouteParallel),
            PlanSource::MeasuredCache,
        ))
    );
    let mlx = MoeProfileRequest::clamped(
        ExecutionPhase::Prefill,
        128,
        32,
        4,
        2_880,
        2_880,
        ClampedMoeStorage::Mlx,
    );
    assert_eq!(cached.lookup_moe(mlx), None);
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn mxfp4_moe_profile_persists_launch_geometry() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = MoeProfileRequest::mxfp4(
        ExecutionPhase::Decode,
        1,
        32,
        4,
        2_880,
        2_880,
        MxFp4MoeStorage::Separate,
        GatedActivation::Silu,
    );
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_moe(request));
    tuner.record_moe(
        request,
        MoeProfileExecution::MxFp4(MxFp4MoeExecution::SingleWarp),
        Duration::from_micros(51),
        Duration::from_millis(2),
    );
    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(
        cached.lookup_moe(request),
        Some((
            MoeProfileExecution::MxFp4(MxFp4MoeExecution::SingleWarp),
            PlanSource::MeasuredCache,
        ))
    );
    let interleaved = MoeProfileRequest::mxfp4(
        ExecutionPhase::Decode,
        1,
        32,
        4,
        2_880,
        2_880,
        MxFp4MoeStorage::Interleaved,
        GatedActivation::Silu,
    );
    assert_eq!(cached.lookup_moe(interleaved), None);
    let unbiased = MoeProfileRequest::mxfp8(
        ExecutionPhase::Decode,
        1,
        64,
        4,
        2_880,
        2_880,
        MxFp8MoeStorage::Separate,
        false,
        GatedActivation::Silu,
    );
    assert_eq!(cached.lookup_moe(unbiased), None);
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn mxfp8_moe_profile_isolates_storage_and_geometry() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = MoeProfileRequest::mxfp8(
        ExecutionPhase::Decode,
        1,
        64,
        4,
        2_880,
        2_880,
        MxFp8MoeStorage::Separate,
        true,
        GatedActivation::Silu,
    );
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_moe(request));
    tuner.record_moe(
        request,
        MoeProfileExecution::MxFp8(MxFp8MoeExecution::FourWarps),
        Duration::from_micros(43),
        Duration::from_millis(2),
    );
    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(
        cached.lookup_moe(request),
        Some((
            MoeProfileExecution::MxFp8(MxFp8MoeExecution::FourWarps),
            PlanSource::MeasuredCache,
        ))
    );
    let interleaved = MoeProfileRequest::mxfp8(
        ExecutionPhase::Decode,
        1,
        64,
        4,
        2_880,
        2_880,
        MxFp8MoeStorage::Interleaved,
        true,
        GatedActivation::Silu,
    );
    assert_eq!(cached.lookup_moe(interleaved), None);
    fs::remove_dir_all(&directory)?;
    Ok(())
}
