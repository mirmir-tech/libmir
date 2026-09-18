use runtime::{
    backend::{DecodeRequest, ModelHandle, PrefillRequest, SamplingLogits},
    kv::BlockTable,
};

use super::{
    super::{Sequence, prefix::PrefixReuse},
    ScheduleMode, schedule,
};
use crate::{
    CudaBackend, CudaPrefillSchedule, Result,
    engine::{execution::Output, model::GenerationExecution},
};

struct Runner(CudaPrefillSchedule);
impl GenerationExecution for Runner {
    fn prefill_schedule(&self) -> CudaPrefillSchedule {
        self.0
    }

    fn prefill_chunk_len(&self, remaining: usize) -> usize {
        remaining.min(1024)
    }

    fn prefill_chunk(
        &mut self,
        _: &CudaBackend,
        _: &PrefillRequest,
        _: &[u32],
        _: usize,
        _: &BlockTable,
        _: bool,
    ) -> Result<Option<Output>> {
        unreachable!()
    }

    fn decode(&mut self, _: &CudaBackend, _: &DecodeRequest, _: bool) -> Result<Output> {
        unreachable!()
    }

    fn clear_sessions(&mut self) {}

    fn release_session(&mut self, _: uuid::Uuid) {}
}
fn row(tokens: usize) -> Sequence {
    Sequence::new(
        PrefillRequest {
            model: ModelHandle {
                id: "model".into(),
                backend: "cuda".into(),
            },
            session_id: uuid::Uuid::new_v4(),
            prompt_tokens: vec![1; tokens],
            cache_checkpoints: vec![],
            block_table: BlockTable::with_block_size(16),
            cached_tokens: 0,
            generation_tokens: None,
            sampling_logits: SamplingLogits::None,
        },
        PrefixReuse::default(),
        std::time::Duration::ZERO,
    )
}
#[test]
fn completion_first_carries_budget_and_preserves_declared_checkpoint() -> Result<()> {
    let mut rows = [row(128), row(8192)];
    let runner = Runner(CudaPrefillSchedule::CompletionFirst);
    let chunks = schedule(&runner, &mut rows, 1, 1024, ScheduleMode::PrefillOnly)?;
    assert_eq!(
        chunks.iter().map(|c| (c.row, c.count)).collect::<Vec<_>>(),
        [(0, 128), (1, 896)]
    );
    rows[0].request.cache_checkpoints = vec![64];
    let chunks = schedule(&runner, &mut rows, 1, 1024, ScheduleMode::PrefillOnly)?;
    assert_eq!(chunks[0].count, 64);
    assert!(!chunks[0].final_chunk);
    Ok(())
}
#[test]
fn other_runners_keep_rotation_and_fair_budget() -> Result<()> {
    let mut rows = [row(8192), row(8192)];
    let chunks = schedule(
        &Runner(CudaPrefillSchedule::RoundRobin),
        &mut rows,
        1,
        1024,
        ScheduleMode::PrefillOnly,
    )?;
    assert_eq!(
        chunks.iter().map(|c| (c.row, c.count)).collect::<Vec<_>>(),
        [(1, 512), (0, 512)]
    );
    Ok(())
}
