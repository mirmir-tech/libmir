use mircuda::{FmhaBf16Plan, FmhaBf16Spec};
use runtime::kv::KvCacheDType;

use super::DecodeAttentionConfig;
use crate::{CudaBackend, Result};

pub(super) fn prepare_varlen_fmha(
    backend: &CudaBackend,
    config: DecodeAttentionConfig,
) -> Result<Option<FmhaBf16Plan>> {
    let supported = matches!(config.cache.cache.dtype, KvCacheDType::Auto | KvCacheDType::BFloat16)
        && matches!(config.cache.key_head_dim, 64 | 128 | 256)
        && config.cache.value_head_dim == config.cache.key_head_dim;
    if !supported {
        return Ok(None);
    }
    let spec = FmhaBf16Spec::new(
        config.query_heads,
        config.cache.kv_heads,
        config.cache.key_head_dim,
        config.cache.value_head_dim,
    )?;
    Ok(Some(FmhaBf16Plan::new(&backend.inner.context, &backend.inner.stream, spec)?))
}
