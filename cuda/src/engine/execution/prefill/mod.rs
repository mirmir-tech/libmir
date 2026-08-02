use std::time::{Duration, Instant};

use runtime::{
    backend::{PrefillOutput, PrefillRequest},
    kv::BlockTable,
    progress::ProgressEvent,
};

use super::Output;
use crate::{Error, Result, engine::CudaEngine};

mod plan;
mod prefix;
mod profile;
mod round;
mod sequence;
#[cfg(test)]
mod tests;

pub struct CudaPrefillBatch {
    model_id: String,
    sequences: Vec<Sequence>,
    cursor: usize,
    rounds: usize,
    token_budget: usize,
    scheduled_tokens: usize,
    started: Instant,
}

struct Sequence {
    request: PrefillRequest,
    consumed: usize,
    chunks: usize,
    prefix_tokens: usize,
    checkpoint_restored: bool,
    runner_wait: Duration,
    completed_at: Option<Instant>,
    step_table: BlockTable,
    output: Option<Output>,
}

impl CudaEngine {
    pub fn prefill_with_progress(
        &self,
        request: &PrefillRequest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let mut indexed = |_: usize, event| progress(event);
        let mut outputs = self.prefill_batch_with_progress(
            std::slice::from_ref(request),
            usize::MAX,
            &mut indexed,
        )?;
        outputs.pop().ok_or(Error::InvalidDecoderKernel("CUDA prefill batch is empty"))
    }

    pub fn prefill_batch_with_progress(
        &self,
        requests: &[PrefillRequest],
        token_budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<Vec<PrefillOutput>> {
        let mut batch = self.prepare_prefill_batch(requests, progress)?;
        while !self.execute_prefill_batch_step(&mut batch, token_budget, progress)? {}
        self.finish_prefill_batch(batch)
    }

    pub fn prepare_prefill_batch(
        &self,
        requests: &[PrefillRequest],
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<CudaPrefillBatch> {
        let first = requests
            .first()
            .ok_or(Error::InvalidDecoderKernel("CUDA prefill batch is empty"))?;
        validate_requests(requests, first)?;
        let loaded = self.model(&first.model.id)?;
        let started = Instant::now();
        let prefix_started = Instant::now();
        let mut runner = loaded.prefill_runner()?;
        let sequences = requests
            .iter()
            .cloned()
            .map(|request| {
                let prefix = prefix::prepare(&mut runner.execution, &request)?;
                Ok(Sequence::new(request, prefix, prefix_started.elapsed()))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(runner);
        for (row, sequence) in sequences.iter().enumerate() {
            progress(
                row,
                ProgressEvent::prefill_tokens(
                    sequence.consumed,
                    sequence.request.prompt_tokens.len(),
                ),
            );
        }
        let scheduled_tokens = sequences
            .iter()
            .map(|sequence| sequence.request.prompt_tokens.len() - sequence.consumed)
            .sum::<usize>();
        Ok(CudaPrefillBatch {
            model_id: first.model.id.clone(),
            sequences,
            cursor: 0,
            rounds: 0,
            token_budget: 0,
            scheduled_tokens,
            started,
        })
    }

    pub fn execute_prefill_batch_step(
        &self,
        batch: &mut CudaPrefillBatch,
        token_budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<bool> {
        if !batch.sequences.iter().any(Sequence::pending) {
            return Ok(true);
        }
        let loaded = self.model(&batch.model_id)?;
        let budget = token_budget.max(1);
        let waiting = Instant::now();
        let mut runner = loaded.prefill_runner()?;
        let wait = waiting.elapsed();
        self.execute_prefill_batch_step_with_runner(
            batch, budget, false, progress, &mut runner, wait,
        )
    }

    pub(super) fn execute_prefill_batch_step_with_runner(
        &self,
        batch: &mut CudaPrefillBatch,
        token_budget: usize,
        interleaved_decode: bool,
        progress: &mut dyn FnMut(usize, ProgressEvent),
        runner: &mut crate::engine::model::ModelRunner,
        runner_wait: Duration,
    ) -> Result<bool> {
        if !batch.sequences.iter().any(Sequence::pending) {
            return Ok(true);
        }
        let budget = token_budget.max(1);
        let rows = self.execute_prefill_round_with_runner(
            runner,
            &mut batch.sequences,
            batch.cursor,
            budget,
            interleaved_decode,
            runner_wait,
        )?;
        batch.rounds += 1;
        batch.token_budget = budget;
        for row in rows {
            batch.cursor = (row + 1) % batch.sequences.len();
            progress(
                row,
                ProgressEvent::prefill_tokens(
                    batch.sequences[row].consumed,
                    batch.sequences[row].request.prompt_tokens.len(),
                ),
            );
        }
        std::thread::yield_now();
        Ok(!batch.sequences.iter().any(Sequence::pending))
    }

    pub(super) fn prefill_batch_model_id(batch: &CudaPrefillBatch) -> &str {
        &batch.model_id
    }

    pub fn finish_prefill_batch(&self, batch: CudaPrefillBatch) -> Result<Vec<PrefillOutput>> {
        if batch.sequences.iter().any(Sequence::pending) {
            return Err(Error::State("CUDA prefill batch is still pending".into()));
        }
        let loaded = self.model(&batch.model_id)?;
        tracing::debug!(
            rows = batch.sequences.len(),
            rounds = batch.rounds,
            token_budget = batch.token_budget,
            scheduled_tokens = batch.scheduled_tokens,
            "completed token-budgeted CUDA prefill batch"
        );
        batch
            .sequences
            .into_iter()
            .map(|sequence| sequence.finish(&loaded, batch.started))
            .collect()
    }
}

fn validate_requests(requests: &[PrefillRequest], first: &PrefillRequest) -> Result<()> {
    if requests.iter().any(|request| request.prompt_tokens.is_empty()) {
        return Err(Error::InvalidDecoderKernel("CUDA prefill prompt is empty"));
    }
    if requests.iter().any(|request| {
        request.model.id != first.model.id || request.model.backend != first.model.backend
    }) {
        return Err(Error::InvalidDecoderKernel("CUDA prefill batch targets multiple models"));
    }
    Ok(())
}
