//! Concurrency warm-up: every decode batch shape and the cost of a session,
//! measured before the K/V budget is.

use runtime::backend::{ModelHandle, PrefillRequest, SamplingLogits};
use uuid::Uuid;

use super::MixedMixerExecution;
use crate::{CudaBackend, Result, engine::model::generation::PrefillChunk};

/// Prefill chunks per throwaway session: two exercise the reusable-prefix
/// path, as the profile warm-up does.
const WARM_SESSION_CHUNKS: usize = 2;

/// Prefills `rows` throwaway sessions through the traffic prefill path, so
/// they reuse its retained chunk shape instead of building a second prefill
/// workspace, then decodes them as
/// batches of every row count from `rows` down to one, then drops them. The
/// batches and what they allocate on first execution stay resident for the
/// memory measurement; the pool bytes one live session costs are recorded
/// for the budget's traffic slack.
pub(super) fn warm_concurrency(
    execution: &mut MixedMixerExecution,
    backend: &CudaBackend,
    rows: usize,
) -> Result<()> {
    let block_size = execution.template.cache_config().block_size;
    let chunk = execution.prefill_chunk_tokens;
    let tokens = chunk.saturating_mul(WARM_SESSION_CHUNKS);
    // Every row decodes once per batch width, so the tables grow by `rows`.
    let blocks_per_session =
        tokens.saturating_add(rows).div_ceil(block_size.max(1)).saturating_add(1);
    let prompt = vec![1_u32; tokens];
    let model = ModelHandle {
        id: "warm-concurrency".into(),
        backend: "cuda".into(),
    };
    let used_before = execution.template.pool_used_bytes()?;
    let mut sequences = Vec::with_capacity(rows);
    let result = (|| {
        for row in 0..rows {
            let session_id = Uuid::new_v4();
            let mut table = runtime::kv::BlockTable::with_block_size(block_size);
            let first_block = row.saturating_mul(blocks_per_session);
            for block in first_block..first_block.saturating_add(blocks_per_session) {
                table.push(runtime::kv::BlockId(u32::try_from(block)?));
            }
            table.set_token_len(tokens);
            let request = PrefillRequest {
                model: model.clone(),
                session_id,
                prompt_tokens: prompt.clone(),
                cache_checkpoints: Vec::new(),
                block_table: table.clone(),
                cached_tokens: 0,
                generation_tokens: None,
                sampling_logits: SamplingLogits::None,
            };
            for index in 0..WARM_SESSION_CHUNKS {
                let offset = index.saturating_mul(chunk);
                let mut chunk_table = table.clone();
                chunk_table.set_token_len(offset.saturating_add(chunk));
                super::batch::prefill(
                    execution,
                    backend,
                    &[PrefillChunk {
                        request: &request,
                        tokens: &prompt[offset..offset.saturating_add(chunk)],
                        offset,
                        table: &chunk_table,
                        final_chunk: index + 1 == WARM_SESSION_CHUNKS,
                    }],
                )?;
            }
            table.set_token_len(tokens.saturating_add(1));
            sequences.push(runtime::backend::DecodeSequence {
                session_id,
                token_id: 1,
                block_table: table,
                sampling_logits: SamplingLogits::None,
            });
        }
        let used_after = execution.template.pool_used_bytes()?;
        let session_bytes = used_after
            .saturating_sub(used_before)
            .checked_div(u64::try_from(rows)?)
            .unwrap_or(0);
        for count in (1..=rows).rev() {
            super::batch::decode(execution, backend, &sequences[..count])?;
            for sequence in &mut sequences[..count] {
                let len = sequence.block_table.token_len();
                sequence.block_table.set_token_len(len.saturating_add(1));
            }
        }
        tracing::debug!(
            rows,
            session_tokens = tokens,
            session_bytes,
            "warmed CUDA decode batch shapes"
        );
        execution.session_bytes = Some(session_bytes);
        Ok(())
    })();
    for sequence in &sequences {
        execution.sessions.remove(&sequence.session_id);
    }
    execution.checkpoints.clear();
    result
}
