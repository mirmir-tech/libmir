use runtime::kv::KvCacheDType;

use crate::{CudaDenseWeightPolicy, DenseRole};

pub(super) fn cache_dtype() -> KvCacheDType {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_KV_FP8").is_some() {
        KvCacheDType::Fp8E4M3
    } else {
        KvCacheDType::BFloat16
    }
}

pub(super) fn dense_weight() -> Result<CudaDenseWeightPolicy, Box<dyn std::error::Error>> {
    let Some(role) = std::env::var_os("LIBMIR_CUDA_PROFILE_BLOCK_FP8_ROLE") else {
        return Ok(CudaDenseWeightPolicy::Bf16);
    };
    match role.to_str() {
        Some("attention-output") => {
            Ok(CudaDenseWeightPolicy::BlockFp8Role(DenseRole::AttentionOutput))
        },
        Some("attention-output-residual") => {
            Ok(CudaDenseWeightPolicy::Fp8Int4Role(DenseRole::AttentionOutput))
        },
        _ => Err("invalid LIBMIR_CUDA_PROFILE_BLOCK_FP8_ROLE".into()),
    }
}
