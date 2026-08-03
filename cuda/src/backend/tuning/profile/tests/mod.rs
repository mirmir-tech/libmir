use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use mircuda::DeviceInfo;
use uuid::Uuid;

use super::CudaAutoTuner;
use crate::{
    AttentionExecution, AttentionPlanRequest, CudaTuningConfig, CudaTuningMode, DenseExecution,
    DensePlanRequest, DenseRole, ExecutionPhase, GatedActivation, MoeExecution, MoePlanRequest,
    PlanSource,
    backend::tuning::{
        AffineMoeExecution, AttentionFamily, AttentionProfileRequest, ClampedMoeExecution,
        ClampedMoeStorage, MoeProfileExecution, MoeProfileRequest, MxFp4MoeExecution,
        MxFp4MoeStorage, MxFp8MoeExecution, MxFp8MoeStorage,
    },
};

mod moe;
mod quantized;

#[test]
fn measured_profile_survives_as_device_shape_cache() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = request();
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_dense(request));
    tuner.record_dense(
        request,
        DenseExecution::CublasLt,
        Duration::from_micros(125),
        Duration::from_millis(2),
    );
    assert_eq!(
        tuner.lookup_dense(request),
        Some((DenseExecution::CublasLt, PlanSource::MeasuredStartup))
    );

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(
        cached.lookup_dense(request),
        Some((DenseExecution::CublasLt, PlanSource::MeasuredCache))
    );
    let payload = profile_payload(&directory)?;
    assert!(!payload.contains("model"));
    assert!(payload.contains("\"input_features\": 3072"));
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn device_identity_invalidates_profile() -> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = request();
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_dense(request));
    tuner.record_dense(
        request,
        DenseExecution::Vector,
        Duration::from_micros(80),
        Duration::from_millis(1),
    );
    let mut other = device();
    other.multiprocessor_count += 1;
    let cached = CudaAutoTuner::new(&other, config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_dense(request), None);
    fs::remove_dir_all(&directory)?;
    Ok(())
}

#[test]
fn finishing_startup_prevents_new_runtime_measurements() {
    let request = request();
    let moe = MoeProfileRequest::nvfp4(
        MoePlanRequest::nvfp4(ExecutionPhase::Decode, 1, 128, 8, 2_048, 768),
        GatedActivation::GeluTanh,
        false,
    );
    let tuner = CudaAutoTuner::new(&device(), CudaTuningConfig::default());
    assert!(tuner.prepares_candidates(PlanSource::Heuristic));

    tuner.finish_startup();

    assert!(!tuner.prepares_candidates(PlanSource::Heuristic));
    assert!(!tuner.claim_dense(request));
    assert!(!tuner.claim_moe(moe));
}

#[test]
fn attention_profile_survives_with_window_and_storage_geometry()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = temporary_directory();
    let request = AttentionProfileRequest {
        family: AttentionFamily::Paged,
        plan: AttentionPlanRequest {
            max_context_tokens: 100_000,
            query_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            value_head_dim: 128,
        },
        block_size: 16,
        dtype: runtime::kv::KvCacheDType::BFloat16,
        window_tokens: Some(4_096),
    };
    let execution = AttentionExecution::SplitKv {
        partition_tokens: 128,
        threshold_tokens: 512,
    };
    let tuner = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Startup));
    assert!(tuner.claim_attention(request));
    tuner.record_attention(request, execution, Duration::from_micros(42), Duration::from_millis(5));

    let cached = CudaAutoTuner::new(&device(), config(&directory, CudaTuningMode::Cached));
    assert_eq!(cached.lookup_attention(request), Some((execution, PlanSource::MeasuredCache)));
    let payload = profile_payload(&directory)?;
    assert!(payload.contains("\"attention\""));
    assert!(payload.contains("\"window_tokens\": 4096"));
    fs::remove_dir_all(&directory)?;
    Ok(())
}

fn request() -> DensePlanRequest {
    DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: DenseRole::AttentionQkv,
        tokens: 1,
        input_features: 3_072,
        output_features: 7_168,
    }
}

fn device() -> DeviceInfo {
    DeviceInfo {
        ordinal: 0,
        name: "Test Accelerator".into(),
        compute_capability: (12, 1),
        multiprocessor_count: 48,
        total_memory: 64 * 1_024 * 1_024 * 1_024,
        memory_pools: true,
        integrated: true,
    }
}

fn config(directory: &Path, mode: CudaTuningMode) -> CudaTuningConfig {
    CudaTuningConfig {
        mode,
        cache_directory: Some(directory.into()),
        ..CudaTuningConfig::default()
    }
}

fn temporary_directory() -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-tuning-{}", Uuid::new_v4()))
}

fn profile_payload(directory: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let path = fs::read_dir(directory)?.next().ok_or("missing tuning profile")??.path();
    Ok(fs::read_to_string(path)?)
}
