use std::fmt::Debug;

use super::{Array, Error, ImageTokenSpan, Result, Stream};

mod cache;
mod clamped_routed;
mod dense;
mod layers;
mod shared_routed;

pub use cache::DecoderCache;
pub(super) use layers::prefill_evaluation_step;
pub use layers::{
    LayerContext, LayerLoopOptions, LoweredLayer, LoweredPackedLayer, MixerKind, forward_layers,
    forward_packed_layers,
};

pub trait DecoderExecution: Debug + Send {
    fn prefers_packed_decode(&self, _stream: &Stream) -> bool {
        true
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

#[derive(Debug)]
pub struct DecoderModel {
    execution: Box<dyn DecoderExecution>,
}

impl DecoderModel {
    pub(crate) fn new(execution: impl DecoderExecution + 'static) -> Self {
        Self { execution: Box::new(execution) }
    }

    pub(crate) fn prefers_packed_decode(&self, stream: &Stream) -> bool {
        self.execution.prefers_packed_decode(stream)
    }

    pub(crate) fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        self.execution.new_cache(stream)
    }

    pub(crate) fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.execution.forward_decode(token_ids, cache, position, stream)
    }

    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        if caches.len() != positions.len() {
            return Err(Error::InvalidModel("packed cache and position row counts differ".into()));
        }
        self.execution.forward_packed_decode(token_ids, caches, positions, stream)
    }

    pub(crate) fn supports_packed_prefill(&self) -> bool {
        self.execution.supports_packed_prefill()
    }

    pub(crate) fn forward_packed_prefill_state(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        if caches.len() != positions.len() {
            return Err(Error::InvalidModel("packed cache and position row counts differ".into()));
        }
        self.execution
            .forward_packed_prefill_state(token_ids, caches, positions, stream)
    }

    pub(crate) fn forward_greedy_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.execution.forward_greedy_decode(token_ids, cache, position, stream)
    }

    pub(crate) fn forward_prefill_state(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.execution.forward_prefill_state(token_ids, cache, position, stream)
    }

    pub(crate) fn forward_pooled_multimodal(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        bidirectional_image_attention: bool,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        self.execution.forward_pooled_multimodal(
            token_ids,
            image_embeddings,
            image,
            bidirectional_image_attention,
            cache,
            stream,
        )
    }

    pub(crate) fn forward_spatial_multimodal(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        position_ids: &Array,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        self.execution.forward_spatial_multimodal(
            token_ids, image_embeddings, image, position_ids, cache, stream,
        )
    }

    pub(crate) fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.execution.fusion_summary()
    }

    pub(crate) fn expert_fusion_summary(&self) -> String {
        self.execution.expert_fusion_summary()
    }
}

fn unsupported(operation: &str) -> Error {
    Error::InvalidModel(format!("lowered decoder does not support {operation}"))
}
