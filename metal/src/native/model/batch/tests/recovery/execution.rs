use super::*;
use crate::engine::{Array, DecoderCache, DecoderExecution, Error, Result, Stream};

#[derive(Debug)]
pub(super) struct FailAfterWrite {
    pub(super) inner: DecoderModel,
    pub(super) calls: Arc<AtomicUsize>,
    pub(super) timing: FaultTiming,
}

#[derive(Debug)]
pub(super) enum FaultTiming {
    NextForward,
    DuringTuning,
    AfterTuning,
}

impl FailAfterWrite {
    fn should_fail(&self) -> bool {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        call == match self.timing {
            FaultTiming::NextForward | FaultTiming::DuringTuning => 1,
            FaultTiming::AfterTuning => 12,
        }
    }

    fn write_then_fail(cache: &mut DecoderCache, stream: &Stream) -> Result<()> {
        let layer = &mut cache.attention_caches_mut()?[0];
        let before = layer.offset()?;
        let value = Array::from_f32(&[0.0; 16], &[1, 2, 1, 8])?;
        let context = layer.update(&value, &value, stream)?;
        assert_eq!(layer.offset()?, before + 1);
        stream.eval_many_with_paged_arenas(&[&context.keys, &context.values])?;
        stream.synchronize()?;
        Err(Error::InvalidModel("injected failure after evaluated K/V write".into()))
    }
}

impl DecoderExecution for FailAfterWrite {
    fn has_decode_plan_candidates(&self) -> bool {
        matches!(self.timing, FaultTiming::AfterTuning | FaultTiming::DuringTuning)
    }

    fn forward_prefill(
        &self,
        ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.inner.forward_prefill_state(ids, cache, position, stream)
    }

    fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        self.inner.new_cache(stream)
    }

    fn forward_decode(
        &self,
        ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        if self.should_fail() {
            Self::write_then_fail(cache, stream)?;
        }
        self.inner.forward_decode(ids, cache, position, stream)
    }

    fn forward_packed_decode(
        &self,
        ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        if self.should_fail() {
            Self::write_then_fail(caches[0], stream)?;
        }
        self.inner.forward_packed_decode(ids, caches, positions, stream)
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        (0, 0, 0, 0)
    }

    fn expert_fusion_summary(&self) -> String {
        String::new()
    }
}

#[derive(Debug)]
pub(super) struct Empty;
impl DecoderExecution for Empty {
    fn forward_prefill(
        &self,
        _: &Array,
        _: &mut DecoderCache,
        _: i32,
        _: &Stream,
    ) -> Result<Array> {
        unreachable!()
    }

    fn new_cache(&self, _: &Stream) -> Result<DecoderCache> {
        unreachable!()
    }

    fn forward_decode(&self, _: &Array, _: &mut DecoderCache, _: i32, _: &Stream) -> Result<Array> {
        unreachable!()
    }

    fn forward_packed_decode(
        &self,
        _: &Array,
        _: &mut [&mut DecoderCache],
        _: &[i32],
        _: &Stream,
    ) -> Result<Array> {
        unreachable!()
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        unreachable!()
    }

    fn expert_fusion_summary(&self) -> String {
        unreachable!()
    }
}
