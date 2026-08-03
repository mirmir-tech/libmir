use super::DecoderExecution;
use crate::engine::{
    Array, DecoderCache, ImageTokenSpan, Result, Stream, hybrid_linear_moe::HybridLinearMoeModel,
};

impl DecoderExecution for HybridLinearMoeModel {
    fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        self.new_cache(stream)
    }

    fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_decode(token_ids, cache, position, stream)
    }

    fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_packed_decode(token_ids, caches, positions, stream)
    }

    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn forward_packed_prefill_state(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_packed_prefill_state(token_ids, caches, positions, stream)
    }

    fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_prefill(token_ids, cache, position, stream)
    }

    fn forward_spatial_multimodal(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        position_ids: &Array,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_multimodal_prefill(
            token_ids, image_embeddings, image, position_ids, cache, stream,
        )
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.fusion_summary()
    }

    fn expert_fusion_summary(&self) -> String {
        self.expert_fusion_summary()
    }
}
