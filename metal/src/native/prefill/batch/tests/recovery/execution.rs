use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    engine::{Array, DecoderCache, DecoderExecution, DecoderModel, Error, Result, Stream},
    native::model::{LoadedExecution, LoadedModel},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    Prefill,
    Decode,
}

#[derive(Debug)]
struct Fault {
    inner: DecoderModel,
    stage: Stage,
    fail_at: usize,
    calls: Arc<AtomicUsize>,
}

pub(super) fn install(model: &mut LoadedModel, stage: Stage, fail_at: usize) -> Arc<AtomicUsize> {
    let previous = std::mem::replace(
        &mut model.execution,
        LoadedExecution::Generation(DecoderModel::new(Empty)),
    );
    let LoadedExecution::Generation(inner) = previous else {
        unreachable!()
    };
    let calls = Arc::new(AtomicUsize::new(0));
    model.execution = LoadedExecution::Generation(DecoderModel::new(Fault {
        inner,
        stage,
        fail_at,
        calls: Arc::clone(&calls),
    }));
    calls
}

impl Fault {
    fn finish(
        &self,
        stage: Stage,
        output: Array,
        caches: &[&DecoderCache],
        stream: &Stream,
    ) -> Result<Array> {
        if stage == self.stage && self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_at {
            let mut roots = vec![&output];
            for cache in caches {
                cache.extend_graph_roots(&mut roots);
            }
            stream.eval_many_with_paged_arenas(&roots)?;
            stream.synchronize()?;
            return Err(Error::InvalidModel(
                "injected failure after evaluated prefill writes".into(),
            ));
        }
        Ok(output)
    }
}

impl DecoderExecution for Fault {
    fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        self.inner.new_cache(stream)
    }

    fn supports_packed_prefill(&self) -> bool {
        self.inner.supports_packed_prefill()
    }

    fn forward_prefill(
        &self,
        ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let output = self.inner.forward_prefill_state(ids, cache, position, stream)?;
        self.finish(Stage::Prefill, output, &[cache], stream)
    }

    fn forward_packed_prefill_state(
        &self,
        ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        let output = self.inner.forward_packed_prefill_state(ids, caches, positions, stream)?;
        let caches = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
        self.finish(Stage::Prefill, output, &caches, stream)
    }

    fn forward_decode(
        &self,
        ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let output = self.inner.forward_decode(ids, cache, position, stream)?;
        self.finish(Stage::Decode, output, &[cache], stream)
    }

    fn forward_packed_decode(
        &self,
        ids: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array> {
        self.inner.forward_packed_decode(ids, caches, positions, stream)
    }

    fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.inner.fusion_summary()
    }

    fn expert_fusion_summary(&self) -> String {
        self.inner.expert_fusion_summary()
    }
}

#[derive(Debug)]
struct Empty;
impl DecoderExecution for Empty {
    fn new_cache(&self, _: &Stream) -> Result<DecoderCache> {
        unreachable!()
    }

    fn forward_prefill(
        &self,
        _: &Array,
        _: &mut DecoderCache,
        _: i32,
        _: &Stream,
    ) -> Result<Array> {
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
