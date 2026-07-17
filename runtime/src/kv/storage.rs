use super::{CacheConfig, KvCacheDType, KvElementBits, KvScaleGranularity, KvWritePlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    Nhd,
    Hnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvStorageSpec {
    pub cache: CacheConfig,
    pub layout: KvCacheLayout,
    pub kv_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub native_bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMemoryBudget {
    pub data_bytes_per_token: usize,
    pub scale_bytes_per_token: usize,
    pub bytes_per_block: usize,
    pub total_bytes: usize,
}

pub trait KvBackendStorage {
    type Tensor;
    type Error;

    #[must_use]
    fn dtype(&self) -> KvCacheDType;

    fn store(
        &mut self,
        plan: &KvWritePlan,
        keys: &Self::Tensor,
        values: &Self::Tensor,
    ) -> Result<usize, Self::Error>;

    #[must_use]
    fn resident_token_slots(&self) -> usize;
}

impl KvStorageSpec {
    #[must_use]
    pub fn new(cache: CacheConfig, kv_heads: usize, head_dim: usize) -> Self {
        Self {
            cache,
            layout: KvCacheLayout::Nhd,
            kv_heads,
            key_head_dim: head_dim,
            value_head_dim: head_dim,
            native_bits: 16,
        }
    }

    #[must_use]
    pub fn element_bits(self) -> KvElementBits {
        self.cache.dtype.element_bits(self.native_bits)
    }

    #[must_use]
    pub fn memory_budget(self) -> KvMemoryBudget {
        let data_bytes_per_token = self.data_bytes_per_token();
        let scale_bytes_per_token = self.scale_bytes_per_token();
        let bytes_per_block =
            (data_bytes_per_token + scale_bytes_per_token) * self.cache.block_size;
        KvMemoryBudget {
            data_bytes_per_token,
            scale_bytes_per_token,
            bytes_per_block,
            total_bytes: bytes_per_block * self.cache.block_count as usize,
        }
    }

    #[must_use]
    pub fn data_bytes_per_token(self) -> usize {
        let bits = self.element_bits();
        let key_bits = self.kv_heads * self.key_head_dim * usize::from(bits.key);
        let value_bits = self.kv_heads * self.value_head_dim * usize::from(bits.value);
        (key_bits + value_bits).div_ceil(8)
    }

    #[must_use]
    pub fn scale_bytes_per_token(self) -> usize {
        match self.cache.dtype.scale_granularity() {
            KvScaleGranularity::None | KvScaleGranularity::Tensor => 0,
            KvScaleGranularity::TokenHead => 2 * self.kv_heads * size_of::<f32>(),
            KvScaleGranularity::Group => self.group_scale_bytes_per_token(),
        }
    }

    fn group_scale_bytes_per_token(self) -> usize {
        if self.cache.dtype == KvCacheDType::NvFp4 {
            let key_scales = self.kv_heads * self.key_head_dim.div_ceil(16);
            let value_scales = self.kv_heads * self.value_head_dim.div_ceil(16);
            key_scales + value_scales
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_native_fp16_pages() {
        let spec = KvStorageSpec::new(CacheConfig::new(2), 8, 128);
        let budget = spec.memory_budget();

        assert_eq!(budget.data_bytes_per_token, 4096);
        assert_eq!(budget.scale_bytes_per_token, 0);
        assert_eq!(budget.bytes_per_block, 65_536);
        assert_eq!(budget.total_bytes, 131_072);
    }

    #[test]
    fn includes_per_token_head_scales() {
        let spec = KvStorageSpec::new(
            CacheConfig {
                block_size: 16,
                block_count: 1,
                dtype: KvCacheDType::Int4PerTokenHead,
            },
            4,
            128,
        );
        let budget = spec.memory_budget();

        assert_eq!(budget.data_bytes_per_token, 512);
        assert_eq!(budget.scale_bytes_per_token, 32);
    }

    #[test]
    fn estimates_nvfp4_group_scales() {
        let spec = KvStorageSpec::new(
            CacheConfig {
                block_size: 16,
                block_count: 1,
                dtype: KvCacheDType::NvFp4,
            },
            8,
            128,
        );
        let budget = spec.memory_budget();

        assert_eq!(budget.data_bytes_per_token, 1024);
        assert_eq!(budget.scale_bytes_per_token, 128);
    }
}
