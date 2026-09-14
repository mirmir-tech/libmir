use std::time::{Duration, Instant};

use runtime::{backend::DecodeBatchRequest, progress::ProgressEvent};

use super::{CudaPrefillBatch, Sequence, round};
use crate::{
    Error, Result,
    engine::{
        CudaEngine,
        execution::{CudaGenerationStepOutput, output::decode_output},
        model::{ModelExecution, ModelRunner, PrefillChunk},
    },
};

impl CudaEngine {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine) fn try_execute_combined_step(
        &self,
        runner: &mut ModelRunner,
        batch: &mut CudaPrefillBatch,
        decode: &DecodeBatchRequest,
        budget: usize,
        progress: &mut dyn FnMut(usize, ProgressEvent),
        runner_wait: Duration,
    ) -> Result<Option<CudaGenerationStepOutput>> {
        if budget == 0 || !batch.sequences.iter().any(Sequence::pending) {
            return Ok(None);
        }
        let ModelExecution::Generation(generation) = &mut runner.execution else {
            return Ok(None);
        };
        let scheduled = round::schedule(
            generation.as_ref(),
            &mut batch.sequences,
            batch.cursor,
            budget,
            round::ScheduleMode::CombinedDecode,
        )?;
        let chunks = scheduled
            .iter()
            .map(|row| {
                let sequence = &batch.sequences[row.row];
                PrefillChunk {
                    request: &sequence.request,
                    tokens: &sequence.request.prompt_tokens[row.offset..row.offset + row.count],
                    offset: row.offset,
                    table: &row.table,
                    final_chunk: row.final_chunk,
                }
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let Some(outputs) =
            generation.prefill_decode_batch(&self.backend, &chunks, decode.sequences())?
        else {
            return Ok(None);
        };
        let execution = started.elapsed();
        if outputs.prefill.len() != scheduled.len()
            || outputs.decode.len() != decode.sequences().len()
        {
            return Err(Error::InvalidDecoderKernel("combined generation output count differs"));
        }
        runner.selected = None;
        for (row, output) in scheduled.iter().zip(outputs.prefill) {
            let sequence = &mut batch.sequences[row.row];
            sequence.runner_wait += runner_wait;
            sequence.execution += execution;
            sequence.consumed += row.count;
            sequence.chunks += 1;
            if let Some(output) = output {
                sequence.completed_at = Some(Instant::now());
                sequence.output = Some(output);
            }
            batch.cursor = (row.row + 1) % batch.sequences.len();
        }
        batch.rounds += 1;
        batch.token_budget = budget;
        for row in &scheduled {
            let sequence = &batch.sequences[row.row];
            progress(
                row.row,
                ProgressEvent::prefill_tokens(
                    sequence.consumed,
                    sequence.request.prompt_tokens.len(),
                ),
            );
        }
        tracing::debug!(
            decode_rows = decode.sequences().len(),
            prefill_rows = scheduled.len(),
            prefill_tokens = scheduled.iter().map(|row| row.count).sum::<usize>(),
            execution_ms = execution.as_secs_f64() * 1000.0,
            "completed combined CUDA generation forward"
        );
        Ok(Some(CudaGenerationStepOutput {
            decode: outputs.decode.into_iter().map(decode_output).collect(),
            prefill: Ok(!batch.sequences.iter().any(Sequence::pending)),
        }))
    }
}
