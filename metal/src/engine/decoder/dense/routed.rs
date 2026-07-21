use super::super::DecoderExecution;
use crate::engine::{
    Array, DecoderCache, ImageTokenSpan, Result, Stream, hybrid_moe::HybridMoeModel,
};

impl DecoderExecution for HybridMoeModel {
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

    fn forward_greedy_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_greedy_decode(token_ids, cache, position, stream)
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

    fn forward_pooled_multimodal(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        bidirectional_image_attention: bool,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_multimodal_prefill(
            token_ids,
            image_embeddings,
            image,
            bidirectional_image_attention,
            cache,
            stream,
        )
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.fusion_summary()
    }

    fn expert_fusion_summary(&self) -> String {
        self.expert_fusion_summary()
    }
}
