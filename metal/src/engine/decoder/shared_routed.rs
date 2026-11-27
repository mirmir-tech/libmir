use super::DecoderExecution;
use crate::engine::{
    Array, DecoderCache, ImageTokenSpan, Result, Stream, hybrid::linear_moe::HybridLinearMoeModel,
};

impl DecoderExecution for HybridLinearMoeModel {
    #[cfg(test)]
    fn prefill_logits(&self, last_hidden: &Array, stream: &Stream) -> Result<Array> {
        self.project_prefill_output(last_hidden, stream)
    }

    fn has_decode_plan_candidates(&self) -> bool {
        self.has_decode_plan_candidates()
    }

    fn keeps_persistent_row_order(&self, stream: &Stream) -> bool {
        stream.config().batch.dense_history.persistent()
    }

    fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        self.new_cache(stream)
    }

    fn new_prediction_cache(&self, stream: &Stream) -> Result<Option<DecoderCache>> {
        self.new_prediction_cache(stream)
    }

    fn forward_prediction(
        &self,
        hidden: &Array,
        next_tokens: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_prediction(hidden, next_tokens, cache, position, stream)
    }

    fn project_hidden(&self, hidden: &Array, stream: &Stream) -> Result<Array> {
        self.project_logits(hidden, stream)
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

    // Speculating sessions prefill one at a time, feeding the prediction
    // module every prompt row.
    fn supports_packed_prefill(&self) -> bool {
        self.prediction.is_none()
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
