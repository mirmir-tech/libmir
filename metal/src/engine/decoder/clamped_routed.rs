use super::DecoderExecution;
use crate::engine::{Array, DecoderCache, Result, Stream, clamped_routed::ClampedRoutedModel};

impl DecoderExecution for ClampedRoutedModel {
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
        self.forward(token_ids, cache, position, false, stream)
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

    fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward(token_ids, cache, position, true, stream)
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        (0, 0, 0, 0)
    }

    fn expert_fusion_summary(&self) -> String {
        "native interleaved MXFP4 expert kernels are enabled".into()
    }
}
