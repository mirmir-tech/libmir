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

    fn prefill_chunk_len(&self, remaining: usize) -> usize;

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

    fn clear_sessions(&mut self);

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
