use std::fmt::Debug;

use super::{Array, DecoderCache, Error, ImageTokenSpan, Result, Stream};

pub trait DecoderExecution: Debug + Send {
    #[cfg(test)]
    fn prefill_logits(&self, _last_hidden: &Array, _stream: &Stream) -> Result<Array> {
        Err(unsupported("terminal prefill logits"))
    }

    fn prefers_packed_decode(&self, _stream: &Stream) -> bool {
        true
    }

    fn has_decode_plan_candidates(&self) -> bool {
        false
    }

    fn new_cache(&self, stream: &Stream) -> Result<DecoderCache>;

    fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array>;

    fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array>;

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn forward_packed_prefill_state(
        &self,
        _token_ids: &Array,
        _caches: &mut [&mut DecoderCache],
        _positions: &[i32],
        _stream: &Stream,
    ) -> Result<Array> {
        Err(unsupported("packed prefill"))
    }

    fn forward_greedy_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_decode(token_ids, cache, position, stream)
    }

    fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array>;

    fn forward_prefill_state(
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
        _token_ids: &Array,
        _image_embeddings: &Array,
        _image: ImageTokenSpan,
        _bidirectional_image_attention: bool,
        _cache: &mut DecoderCache,
        _stream: &Stream,
    ) -> Result<Array> {
        Err(unsupported("pooled multimodal prefill"))
    }

    fn forward_spatial_multimodal(
        &self,
        _token_ids: &Array,
        _image_embeddings: &Array,
        _image: ImageTokenSpan,
        _position_ids: &Array,
        _cache: &mut DecoderCache,
        _stream: &Stream,
    ) -> Result<Array> {
        Err(unsupported("spatial multimodal prefill"))
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize);
    fn expert_fusion_summary(&self) -> String;
}

fn unsupported(operation: &str) -> Error {
    Error::InvalidModel(format!("lowered decoder does not support {operation}"))
}
