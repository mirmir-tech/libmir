use super::{PagedAttentionSpec, SplitPagedAttention};

impl SplitPagedAttention {
    pub(super) fn threads(&self) -> u32 {
        if self.grouped_queries {
            let minimum_warps = if self.tensor_queries {
                4
            } else {
                1
            };
            u32::try_from(self.spec.query_heads / self.spec.kv_heads)
                .map_or(256, |query_group| query_group.max(minimum_warps) * 32)
        } else if self.spec.head_dim <= 128 && self.spec.value_head_dim <= 128 {
            128
        } else {
            256
        }
    }

    pub(super) const fn split_heads(&self) -> usize {
        if self.grouped_queries {
            self.spec.kv_heads
        } else {
            self.spec.query_heads
        }
    }
}

pub(in crate::kernels::paged) fn tensor_queries(
    spec: PagedAttentionSpec,
    query_group: usize,
    grouped_queries: bool,
) -> bool {
    grouped_queries
        && query_group >= 4
        && spec.head_dim == 64
        && spec.value_head_dim == 64
        && matches!(
            spec.dtype,
            runtime::kv::KvCacheDType::Auto | runtime::kv::KvCacheDType::BFloat16
        )
}

#[cfg(test)]
mod tests {
    use runtime::kv::KvCacheDType;

    use super::*;

    #[test]
    fn tensor_queries_accepts_auto_and_explicit_bf16() {
        let spec = PagedAttentionSpec {
            block_size: 16,
            max_blocks: 256,
            query_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            value_head_dim: 64,
            dtype: KvCacheDType::Auto,
        };
        assert!(tensor_queries(spec, 8, true));
        assert!(tensor_queries(
            PagedAttentionSpec { dtype: KvCacheDType::BFloat16, ..spec },
            8,
            true
        ));
        assert!(!tensor_queries(
            PagedAttentionSpec { dtype: KvCacheDType::Fp8E4M3, ..spec },
            8,
            true
        ));
    }
}
