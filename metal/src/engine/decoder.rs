use super::{
    Array, DecoderCache, Result, Stream, dense_swiglu::DenseSwiGluModel,
    hybrid_linear_moe::HybridLinearMoeModel, hybrid_moe::HybridMoeModel,
};

#[derive(Debug)]
pub enum DecoderModel {
    HybridMoe(HybridMoeModel),
    HybridLinearMoe(HybridLinearMoeModel),
    DenseSwiGlu(DenseSwiGluModel),
}

impl DecoderModel {
    pub(crate) fn prefers_packed_decode(&self, stream: &Stream) -> bool {
        match self {
            Self::DenseSwiGlu(_) => stream.config().batch.dense_decode.packed(),
            Self::HybridMoe(_) | Self::HybridLinearMoe(_) => true,
        }
    }

    pub(crate) fn new_cache(&self) -> Result<DecoderCache> {
        match self {
            Self::HybridMoe(model) => model.new_cache(),
            Self::HybridLinearMoe(model) => model.new_cache(),
            Self::DenseSwiGlu(model) => model.new_cache(),
        }
    }

    pub(crate) fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::HybridMoe(model) => model.forward_decode(token_ids, cache, position, stream),
            Self::HybridLinearMoe(model) => {
                model.forward_decode(token_ids, cache, position, stream)
            },
            Self::DenseSwiGlu(model) => model.forward_decode(token_ids, cache, position, stream),
        }
    }

    pub(crate) fn forward_packed_decode(
        &self,
        token_ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        if caches.len() != positions.len() {
            return Err(super::Error::InvalidModel(
                "packed cache and position row counts differ".into(),
            ));
        }
        match self {
            Self::DenseSwiGlu(model) => {
                model.forward_packed_decode(token_ids, caches, positions, stream)
            },
            Self::HybridMoe(model) => {
                model.forward_packed_decode(token_ids, caches, positions, stream)
            },
            Self::HybridLinearMoe(model) => {
                model.forward_packed_decode(token_ids, caches, positions, stream)
            },
        }
    }

    pub(crate) fn forward_greedy_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::HybridMoe(model) => {
                model.forward_greedy_decode(token_ids, cache, position, stream)
            },
            Self::HybridLinearMoe(model) => {
                model.forward_decode(token_ids, cache, position, stream)
            },
            Self::DenseSwiGlu(model) => model.forward_decode(token_ids, cache, position, stream),
        }
    }

    pub(crate) fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::HybridMoe(model) => model.forward_prefill(token_ids, cache, position, stream),
            Self::HybridLinearMoe(model) => {
                model.forward_prefill(token_ids, cache, position, stream)
            },
            Self::DenseSwiGlu(model) => model.forward_prefill(token_ids, cache, position, stream),
        }
    }

    pub(crate) fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        match self {
            Self::HybridMoe(model) => model.fusion_summary(),
            Self::HybridLinearMoe(model) => model.fusion_summary(),
            Self::DenseSwiGlu(model) => {
                let (attention, gate_up) = model.fusion_summary();
                (attention, 0, gate_up, 0)
            },
        }
    }

    pub(crate) fn expert_fusion_summary(&self) -> String {
        match self {
            Self::HybridMoe(model) => model.expert_fusion_summary(),
            Self::HybridLinearMoe(model) => model.expert_fusion_summary(),
            Self::DenseSwiGlu(_) => {
                "expert gate/up fusion is not applicable to dense SwiGLU".into()
            },
        }
    }
}
