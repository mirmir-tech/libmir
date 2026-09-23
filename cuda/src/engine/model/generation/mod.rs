use mircuda::{DeviceBuffer, bf16};
use runtime::{
    backend::{DecodeRequest, DecodeSequence, PrefillRequest},
    kv::BlockTable,
};
use uuid::Uuid;

use crate::{CudaBackend, Error, Result, engine::execution::Output};

mod graph;
mod session;

pub(in crate::engine) use graph::GraphExecution;
pub(in crate::engine) use session::{MixedMixerExecution, SinkAttentionExecution};

pub(in crate::engine) trait GenerationExecution: Send {
    fn prefill_schedule(&self) -> crate::CudaPrefillSchedule {
        crate::CudaPrefillSchedule::RoundRobin
    }

    fn prefix_replay_tokens(&self) -> Option<usize> {
        None
    }

    // Admission may assume a retained checkpoint; replay remains the
    // correctness fallback when that checkpoint is absent.
    fn prefix_admission(&self) -> Option<(usize, usize, usize)> {
        self.prefix_replay_tokens().map(|replay| (replay, replay, 0))
    }

    fn restore_prefix(
        &mut self,
        _request: &PrefillRequest,
        _minimum: usize,
        _maximum: usize,
    ) -> Result<Option<usize>> {
        Ok(None)
    }

    fn terminal_cache_checkpoint(&self, _request: &PrefillRequest) -> Option<usize> {
        None
    }

    fn cache_checkpoint_alignment(&self) -> Option<usize> {
        None
    }

    /// Whether prefill chunks must end at the terminal checkpoint. Executions
    /// that capture it inside a pass let the prompt tail join the last chunk.
    fn splits_at_terminal_checkpoint(&self) -> bool {
        true
    }

    fn prefill_chunk_len(&self, remaining: usize) -> usize;
    fn interleaved_prefill_budget(&self, budget: usize) -> usize {
        budget
    }

    fn prefill_chunk(
        &mut self,
        backend: &CudaBackend,
        request: &PrefillRequest,
        tokens: &[u32],
        offset: usize,
        table: &BlockTable,
        final_chunk: bool,
    ) -> Result<Option<Output>>;

    fn prefill_batch_chunk(
        &mut self,
        backend: &CudaBackend,
        chunks: &[PrefillChunk<'_>],
    ) -> Result<Vec<Option<Output>>> {
        chunks
            .iter()
            .map(|chunk| {
                self.prefill_chunk(
                    backend,
                    chunk.request,
                    chunk.tokens,
                    chunk.offset,
                    chunk.table,
                    chunk.final_chunk,
                )
            })
            .collect()
    }

    fn decode(
        &mut self,
        backend: &CudaBackend,
        request: &DecodeRequest,
        use_device_token: bool,
    ) -> Result<Output>;

    fn decode_batch(
        &mut self,
        _backend: &CudaBackend,
        _sequences: &[DecodeSequence],
    ) -> Result<Option<Vec<Output>>> {
        Ok(None)
    }

    fn prefill_decode_batch(
        &mut self,
        _backend: &CudaBackend,
        _chunks: &[PrefillChunk<'_>],
        _decode: &[DecodeSequence],
    ) -> Result<Option<CombinedOutputs>> {
        Ok(None)
    }

    fn clear_sessions(&mut self);

    /// Pool bytes the runner's retained execution shapes may still grow by
    /// under traffic, beyond what warm-up built.
    fn retention_headroom_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// Runs one decode step over `rows` throwaway sessions, so the batch shape
    /// and its lazily allocated resources exist before memory is measured.
    fn warm_concurrency(&mut self, _backend: &CudaBackend, _rows: usize) -> Result<()> {
        Ok(())
    }

    /// Pool bytes one live session cost when concurrency was warmed, where
    /// the backend measured it.
    fn session_bytes(&self) -> Option<u64> {
        None
    }

    /// Pool bytes held by retained execution shapes right now.
    fn retained_shape_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// Reallocates the K/V pages for `cache`, dropping every session and
    /// retained prefix state. `Ok(false)` when this runner keeps its pages.
    fn resize_kv_cache(&mut self, _cache: runtime::kv::CacheConfig) -> Result<bool> {
        Ok(false)
    }

    fn release_session(&mut self, session: Uuid);

    fn prefill_pooled_vision(
        &mut self,
        _backend: &CudaBackend,
        _input: PooledVisionPrefill<'_>,
    ) -> Result<Output> {
        Err(unsupported("pooled vision prefill"))
    }

    fn prefill_spatial_vision(
        &mut self,
        _backend: &CudaBackend,
        _input: SpatialVisionPrefill<'_>,
    ) -> Result<Output> {
        Err(unsupported("spatial-merge vision prefill"))
    }
}

pub(in crate::engine) struct PrefillChunk<'a> {
    pub request: &'a PrefillRequest,
    pub tokens: &'a [u32],
    pub offset: usize,
    pub table: &'a BlockTable,
    pub final_chunk: bool,
}

pub(in crate::engine) struct PooledVisionPrefill<'a> {
    pub session: Uuid,
    pub tokens: &'a [u32],
    pub image: &'a DeviceBuffer<bf16>,
    pub image_span: (usize, usize),
    pub bidirectional: bool,
    pub table: &'a BlockTable,
    pub sampling: runtime::backend::SamplingLogits,
}

pub(in crate::engine) struct SpatialVisionPrefill<'a> {
    pub session: Uuid,
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
    pub image: &'a DeviceBuffer<bf16>,
    pub image_span: (usize, usize),
    pub position_delta: i32,
    pub table: &'a BlockTable,
    pub sampling: runtime::backend::SamplingLogits,
}

fn unsupported(operation: &'static str) -> Error {
    Error::MissingCapability {
        operation,
        storage: "lowered CUDA generation execution".into(),
        geometry: "model execution boundary".into(),
        requirement: "the lowered mixer plan must implement this multimodal operation",
    }
}

pub(in crate::engine) struct CombinedOutputs {
    pub(in crate::engine) prefill: Vec<Option<Output>>,
    pub(in crate::engine) decode: Vec<Output>,
}
