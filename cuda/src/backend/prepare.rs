use ::runtime::kv::KvStorageSpec;

use super::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    BatchedPagedAttentionBf16, Bf16Linear, Bf16VectorLinear, CudaBackend, GatedActivation,
    NvFp4Bf16Linear, NvFp4Config, NvFp4Tensors, PagedAttentionBf16, PagedDecodeBatch, PagedKvCache,
    RmsNormBf16, RopeBf16, SelectedAffineGatedBf16Linear, SelectedAffinePairBf16Linear,
    SelectedAffineReduceBf16Linear,
};
use crate::{Result, kernels::RopeSpec};

impl CudaBackend {
    pub fn prepare_bf16_linear(
        &self,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Bf16Linear> {
        Bf16Linear::new(self, tokens, input_features, output_features)
    }

    pub fn prepare_bf16_vector_linear(
        &self,
        input_features: usize,
        output_features: usize,
    ) -> Result<Bf16VectorLinear> {
        Bf16VectorLinear::new(self, input_features, output_features)
    }

    pub fn prepare_rms_norm_bf16(
        &self,
        rows: usize,
        features: usize,
        epsilon: f32,
    ) -> Result<RmsNormBf16> {
        RmsNormBf16::new(self, rows, features, epsilon)
    }

    pub fn prepare_rope_bf16(&self, spec: RopeSpec) -> Result<RopeBf16> {
        RopeBf16::new(self, spec)
    }

    pub fn prepare_paged_kv(&self, layer: usize, storage: KvStorageSpec) -> Result<PagedKvCache> {
        PagedKvCache::new(self, layer, storage)
    }

    pub fn prepare_paged_attention_bf16(
        &self,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
    ) -> Result<PagedAttentionBf16> {
        PagedAttentionBf16::new(self, cache, query_heads, max_blocks)
    }

    pub fn prepare_batched_paged_attention_bf16(
        &self,
        cache: &PagedKvCache,
        query_heads: usize,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<BatchedPagedAttentionBf16> {
        BatchedPagedAttentionBf16::new(self, cache, query_heads, max_blocks, max_batch)
    }

    pub fn prepare_paged_decode_batch(
        &self,
        storage: KvStorageSpec,
        max_blocks: usize,
        max_batch: usize,
    ) -> Result<PagedDecodeBatch> {
        PagedDecodeBatch::new(self, storage, max_blocks, max_batch)
    }

    pub fn prepare_nvfp4_bf16_linear(
        &self,
        tokens: usize,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<NvFp4Bf16Linear> {
        NvFp4Bf16Linear::new(self, tokens, config, tensors)
    }

    pub fn prepare_affine_quantized_bf16_linear(
        &self,
        input_features: usize,
        output_features: usize,
        matrices: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<AffineQuantizedBf16Linear> {
        AffineQuantizedBf16Linear::new(
            self, input_features, output_features, matrices, group_size, bits,
        )
    }

    pub fn prepare_affine_quantized_bf16_qmm(
        &self,
        tokens: usize,
        config: AffineQuantizedConfig,
        matrices: usize,
    ) -> Result<AffineQuantizedBf16Qmm> {
        AffineQuantizedBf16Qmm::new(self, tokens, config, matrices)
    }

    pub fn prepare_selected_affine_pair_bf16_linear(
        &self,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<SelectedAffinePairBf16Linear> {
        SelectedAffinePairBf16Linear::new(self, config.spec()?, expert_count, selected_count)
    }

    pub fn prepare_selected_affine_gated_bf16_linear(
        &self,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
        activation: GatedActivation,
    ) -> Result<SelectedAffineGatedBf16Linear> {
        SelectedAffineGatedBf16Linear::new(
            self,
            config.spec()?,
            expert_count,
            selected_count,
            activation,
        )
    }

    pub fn prepare_batched_selected_affine_gated_bf16_linear(
        &self,
        tokens: usize,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
        activation: GatedActivation,
    ) -> Result<SelectedAffineGatedBf16Linear> {
        SelectedAffineGatedBf16Linear::new_batch(
            self,
            config.spec()?,
            expert_count,
            selected_count,
            tokens,
            activation,
        )
    }

    pub fn prepare_selected_affine_reduce_bf16_linear(
        &self,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<SelectedAffineReduceBf16Linear> {
        SelectedAffineReduceBf16Linear::new(self, config.spec()?, expert_count, selected_count)
    }

    pub fn prepare_batched_selected_affine_reduce_bf16_linear(
        &self,
        tokens: usize,
        config: AffineQuantizedConfig,
        expert_count: usize,
        selected_count: usize,
    ) -> Result<SelectedAffineReduceBf16Linear> {
        SelectedAffineReduceBf16Linear::new_batch(
            self,
            config.spec()?,
            expert_count,
            selected_count,
            tokens,
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.stream.synchronize()?)
    }

    pub fn trim_memory_pool(&self, retain_bytes: usize) -> Result<()> {
        self.synchronize()?;
        Ok(self.inner.pool.trim_to(retain_bytes)?)
    }
}
