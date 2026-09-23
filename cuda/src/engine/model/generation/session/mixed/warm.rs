//! Concurrency warm-up: every decode batch shape and the cost of a session,
//! measured before the K/V budget is.

use uuid::Uuid;

use super::MixedMixerExecution;
use crate::{CudaBackend, Result};

/// Prefill chunks per throwaway session: two exercise the reusable-prefix
/// path, as the profile warm-up does.
const WARM_SESSION_CHUNKS: usize = 2;

/// Prefills `rows` throwaway sessions the way traffic does, decodes them as
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
    let prompt = vec![1_u32; chunk];
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
            let mut session = execution.template.instantiate_with_caches(&execution.caches)?;
            for index in 0..WARM_SESSION_CHUNKS {
                table.set_token_len((index + 1).saturating_mul(chunk));
                session.prefill(session_id, &prompt, &table)?;
            }
            execution.sessions.insert(session_id, session);
            table.set_token_len(tokens.saturating_add(1));
            sequences.push(runtime::backend::DecodeSequence {
                session_id,
                token_id: 1,
                block_table: table,
                sampling_logits: runtime::backend::SamplingLogits::None,
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
