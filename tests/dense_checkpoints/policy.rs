use libmir::RuntimeConfig;
#[cfg(feature = "cuda")]
use libmir::cuda::{
    CudaDenseVectorPolicy, CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaKernelAdmission,
    CudaMoeBatchPolicy, CudaNumericalPolicy, CudaTuningMode, DenseRole,
};

use super::TestResult;

#[cfg(feature = "cuda")]
pub fn configure_cuda_policy(config: &mut RuntimeConfig) -> TestResult<()> {
    config.cuda.tuning.mode = match std::env::var("MIRMIR_DENSE_CUDA_TUNING") {
        Err(std::env::VarError::NotPresent) => CudaTuningMode::Startup,
        Ok(value) if value == "startup" => CudaTuningMode::Startup,
        Ok(value) if value == "cached" => CudaTuningMode::Cached,
        Ok(value) if value == "disabled" => CudaTuningMode::Disabled,
        Ok(value) => return Err(format!("unsupported MIRMIR_DENSE_CUDA_TUNING={value}").into()),
        Err(error) => return Err(error.into()),
    };
    match std::env::var("MIRMIR_DENSE_CUDA_POLICY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("stable") => Ok(()),
        Ok("throughput") => {
            config.cuda.planning.numerical = CudaNumericalPolicy::Throughput;
            config.cuda.planning.admission = CudaKernelAdmission::Experimental;
            config.cuda.planning.dense_vectors = CudaDenseVectorPolicy::Tuned;
            config.cuda.planning.dense_vendor = CudaDenseVendorPolicy::Tuned;
            config.cuda.planning.dense_weights =
                CudaDenseWeightPolicy::BlockFp8Role(DenseRole::DenseGateUp);
            Ok(())
        },
        Ok("nvfp4-w4a16") => {
            config.cuda.planning.moe_batch = CudaMoeBatchPolicy::W4A16;
            Ok(())
        },
        _ => Err("MIRMIR_DENSE_CUDA_POLICY must be stable, throughput, or nvfp4-w4a16".into()),
    }
}

#[cfg(not(feature = "cuda"))]
pub fn configure_cuda_policy(_config: &mut RuntimeConfig) -> TestResult<()> {
    if std::env::var_os("MIRMIR_DENSE_CUDA_POLICY").is_some() {
        Err("MIRMIR_DENSE_CUDA_POLICY requires the CUDA feature".into())
    } else {
        Ok(())
    }
}
