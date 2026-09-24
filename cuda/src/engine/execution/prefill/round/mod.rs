use std::time::{Duration, Instant};

use runtime::kv::BlockTable;

use super::{Sequence, plan};
use crate::{
    Error, Result,
    engine::{
        CudaEngine,
        model::{DeviceToken, ModelExecution, ModelRunner, PrefillChunk},
    },
};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ScheduleMode {
    PrefillOnly,
    SeparateDecode,
    CombinedDecode,
}

pub(super) struct ScheduledChunk {
    pub(super) row: usize,
    pub(super) count: usize,
    pub(super) offset: usize,
    pub(super) table: BlockTable,
    pub(super) final_chunk: bool,
    completion_first: bool,
}

impl CudaEngine {
    pub(super) fn execute_prefill_round_with_runner(
        &self,
        runner: &mut ModelRunner,
        sequences: &mut [Sequence],
        cursor: usize,
        budget: usize,
        interleaved_decode: bool,
        runner_wait: Duration,
    ) -> Result<Vec<usize>> {
        let ModelExecution::Generation(generation) = &mut runner.execution else {
            return Err(Error::State("CUDA task is not a generation runner".into()));
        };
        let scheduled = schedule(
            generation.as_ref(),
            sequences,
            cursor,
            budget,
            if interleaved_decode {
                ScheduleMode::SeparateDecode
            } else {
                ScheduleMode::PrefillOnly
            },
        )?;
        let scheduled_tokens = scheduled.iter().map(|chunk| chunk.count).sum::<usize>();
        let final_rows = scheduled.iter().filter(|chunk| chunk.final_chunk).count();
        let completion_first_rows = scheduled.iter().filter(|chunk| chunk.completion_first).count();
        let maximum_context = scheduled
            .iter()
            .map(|chunk| chunk.offset + chunk.count)
            .max()
            .unwrap_or_default();
        let chunks = scheduled
            .iter()
            .map(|chunk| {
                let sequence = &sequences[chunk.row];
                PrefillChunk {
                    request: &sequence.request,
                    tokens: &sequence.request.prompt_tokens
                        [chunk.offset..chunk.offset + chunk.count],
                    offset: chunk.offset,
                    table: &chunk.table,
                    final_chunk: chunk.final_chunk,
                }
            })
            .collect::<Vec<_>>();
        let execution_started = Instant::now();
        let outputs = generation.prefill_batch_chunk(&self.backend, &chunks)?;
        let execution = execution_started.elapsed();
        if outputs.len() != scheduled.len() {
            return Err(Error::InvalidDecoderKernel("CUDA prefill returned another batch size"));
        }
        for (chunk, output) in scheduled.iter().zip(outputs) {
            let sequence = &mut sequences[chunk.row];
            sequence.runner_wait += runner_wait;
            sequence.execution += execution;
            sequence.consumed += chunk.count;
            sequence.chunks += 1;
            if let Some(output) = output {
                sequence.completed_at = Some(Instant::now());
                runner.selected = output.token.map(|token| DeviceToken {
                    session: sequence.request.session_id,
                    token,
                });
                sequence.output = Some(output);
            }
        }
        tracing::debug!(
            rows = scheduled.len(),
            scheduled_tokens,
            unused_tokens = budget.saturating_sub(scheduled_tokens),
            occupancy_per_mille = scheduled_tokens.saturating_mul(1_000) / budget.max(1),
            final_rows,
            completion_first_rows,
            maximum_context,
            token_budget = budget,
            interleaved_decode,
            runner_wait_ms = runner_wait.as_secs_f64() * 1_000.0,
            execution_ms = execution.as_secs_f64() * 1_000.0,
            "completed CUDA prefill round"
        );
        Ok(scheduled.into_iter().map(|chunk| chunk.row).collect())
    }
}

pub(super) fn schedule(
    generation: &dyn crate::engine::model::GenerationExecution,
    sequences: &mut [Sequence],
    cursor: usize,
    budget: usize,
    mode: ScheduleMode,
) -> Result<Vec<ScheduledChunk>> {
    let mut remaining_budget = budget;
    let mut scheduled = Vec::new();
    let fifo = generation.prefill_schedule() == crate::CudaPrefillSchedule::CompletionFirst;
    let rows = round_rows(
        sequences,
        if fifo {
            0
        } else {
            cursor
        },
    );
    for (index, row) in rows.iter().copied().enumerate() {
        let sequence = &mut sequences[row];
        let remaining = sequence.request.prompt_tokens.len() - sequence.consumed;
        let rows_left = rows.len() - index;
        let completion_first =
            fifo || (mode != ScheduleMode::PrefillOnly && sequence.checkpoint_restored);
        let row_budget = plan::row_chunk_budget(remaining_budget, rows_left, completion_first);
        let row_budget = if mode == ScheduleMode::SeparateDecode {
            generation.interleaved_prefill_budget(row_budget)
        } else {
            row_budget
        };
        let context_budget = if mode == ScheduleMode::CombinedDecode {
            // Decode participates in this forward; splitting a configured 2048
            // step into 1024 + decode rows loses its reusable projection shape.
            remaining_budget
        } else {
            plan::context_chunk_budget(
                sequence.consumed,
                rows.len(),
                budget,
                mode != ScheduleMode::PrefillOnly,
                sequence.prefix_tokens > 0,
                completion_first,
            )
        };
        let terminal = generation
            .splits_at_terminal_checkpoint()
            .then(|| generation.terminal_cache_checkpoint(&sequence.request))
            .flatten();
        let alignment = generation.cache_checkpoint_alignment();
        let boundary = sequence.checkpoint_distance(terminal, alignment);
        let limit = remaining.min(row_budget).min(context_budget).min(boundary);
        // Only a fair share is rounded to blocks; a row given the rest of the
        // budget keeps it whole, or completion-first steps lose their shape.
        let fair_share = row_budget < remaining_budget && limit == row_budget;
        let limit = if fair_share {
            plan::aligned_chunk(limit, remaining)
        } else {
            limit
        };
        let absorbed = plan::absorb_remainder(limit, remaining, boundary);
        let mut count = generation.prefill_chunk_len(absorbed);
        // A backend that rounds chunks to its own shapes may hand back
        // neither the whole remainder nor a chunk within the budget; the
        // remainder then stays a chunk of its own.
        if absorbed != limit && count != remaining {
            count = generation.prefill_chunk_len(limit);
        }
        if !plan::valid_chunk(count, remaining, remaining_budget) {
            return Err(Error::InvalidDecoderKernel(
                "CUDA lowering returned an invalid prefill chunk",
            ));
        }
        sequence.step_table.set_token_len(sequence.consumed + count);
        scheduled.push(ScheduledChunk {
            row,
            count,
            offset: sequence.consumed,
            table: sequence.step_table.clone(),
            final_chunk: sequence.consumed + count == sequence.request.prompt_tokens.len(),
            completion_first,
        });
        remaining_budget = remaining_budget.saturating_sub(count);
        if remaining_budget == 0 {
            break;
        }
    }
    Ok(scheduled)
}

fn round_rows(sequences: &[Sequence], cursor: usize) -> Vec<usize> {
    let pending = sequences.iter().map(Sequence::pending).collect::<Vec<_>>();
    plan::round_rows_from_pending(&pending, cursor)
}

#[cfg(test)]
mod tests;
