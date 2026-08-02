use super::*;
use crate::backend::tuning::{
    AffineProjectionExecution, DirectFp8ProjectionExecution, DirectFp8ScaleDType,
    DirectFp8WeightScale, MxFp8ProjectionExecution, QuantizedProfileExecution,
    QuantizedProfileRequest,
};

#[test]
fn affine_projection_profile_survives_with_physical_geometry()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = QuantizedProfileRequest::affine(1, 2_880, 12_288, 64, 4);
    let execution = QuantizedProfileExecution::Affine(AffineProjectionExecution::Gemv);
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_quantized(request));
    tuner.record_quantized(request, execution, Duration::from_micros(81), Duration::from_millis(3));

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_quantized(request), Some((execution, PlanSource::MeasuredCache)));
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::affine(1, 2_880, 12_288, 32, 4)),
        None
    );
    let payload = profile_payload(&directory)?;
    assert!(payload.contains("\"quantized\""));
    assert!(payload.contains("\"group_size\": 64"));
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn direct_fp8_profile_isolated_by_scale_dtype_and_bias() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = QuantizedProfileRequest::direct_fp8_dynamic_e4m3(
        64,
        896,
        4_864,
        DirectFp8ScaleDType::F32,
        false,
    );
    let execution = QuantizedProfileExecution::DirectFp8(DirectFp8ProjectionExecution::TensorCore);
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_quantized(request));
    tuner.record_quantized(request, execution, Duration::from_micros(52), Duration::from_millis(3));

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_quantized(request), Some((execution, PlanSource::MeasuredCache)));
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::direct_fp8_dynamic_e4m3(
            64,
            896,
            4_864,
            DirectFp8ScaleDType::Bf16,
            false,
        )),
        None
    );
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::direct_fp8_dynamic_e4m3(
            64,
            896,
            4_864,
            DirectFp8ScaleDType::F32,
            true,
        )),
        None
    );
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn static_fp8_profile_isolated_by_weight_scale() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = QuantizedProfileRequest::direct_fp8_static_e4m3(
        64,
        896,
        4_864,
        DirectFp8WeightScale::Tensor,
        DirectFp8ScaleDType::Bf16,
        false,
    );
    let execution = QuantizedProfileExecution::DirectFp8(DirectFp8ProjectionExecution::TensorCore);
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_quantized(request));
    tuner.record_quantized(request, execution, Duration::from_micros(52), Duration::from_millis(3));

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_quantized(request), Some((execution, PlanSource::MeasuredCache)));
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::direct_fp8_static_e4m3(
            64,
            896,
            4_864,
            DirectFp8WeightScale::OutputChannel,
            DirectFp8ScaleDType::Bf16,
            false,
        )),
        None
    );
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::direct_fp8_dynamic_e4m3(
            64,
            896,
            4_864,
            DirectFp8ScaleDType::Bf16,
            false,
        )),
        None
    );
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn e5m2_weight_only_profile_isolated_from_e4m3_and_bias() -> Result<(), Box<dyn std::error::Error>>
{
    let directory = temporary_directory();
    let request =
        QuantizedProfileRequest::direct_fp8_bf16_e5m2_weight_only(64, 2_048, 5_632, false);
    let execution = QuantizedProfileExecution::DirectFp8(DirectFp8ProjectionExecution::TensorCore);
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_quantized(request));
    tuner.record_quantized(request, execution, Duration::from_micros(48), Duration::from_millis(3));

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_quantized(request), Some((execution, PlanSource::MeasuredCache)));
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::direct_fp8_bf16_e5m2_weight_only(
            64, 2_048, 5_632, true,
        )),
        None
    );
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::direct_fp8_dynamic_e4m3(
            64,
            2_048,
            5_632,
            DirectFp8ScaleDType::F32,
            false,
        )),
        None
    );
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn mxfp8_profile_isolated_from_affine_and_token_shape() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = QuantizedProfileRequest::mxfp8(64, 1_024, 3_072);
    let execution = QuantizedProfileExecution::MxFp8(MxFp8ProjectionExecution::TensorCore);
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_quantized(request));
    tuner.record_quantized(request, execution, Duration::from_micros(44), Duration::from_millis(2));

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_quantized(request), Some((execution, PlanSource::MeasuredCache)));
    assert_eq!(cached.lookup_quantized(QuantizedProfileRequest::mxfp8(32, 1_024, 3_072)), None);
    assert_eq!(
        cached.lookup_quantized(QuantizedProfileRequest::affine(64, 1_024, 3_072, 32, 8)),
        None
    );
    fs::remove_dir_all(&directory)?;
    Ok(())
}
