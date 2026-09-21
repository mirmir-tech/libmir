use runtime::backend::{DecodeSequence, SamplingLogits};

mod cache;
pub(super) use cache::CombinedBatches;

use super::{
    MixedMixerExecution,
    batch::{restore_sessions, take_sessions},
};
use crate::{
    CudaBackend, Error, Result,
    engine::{
        execution::{Output, device_sampling},
        model::generation::{CombinedOutputs, PrefillChunk},
    },
};

pub(super) fn execute(
    execution: &mut MixedMixerExecution,
    _backend: &CudaBackend,
    chunks: &[PrefillChunk<'_>],
    decode: &[DecodeSequence],
) -> Result<Option<CombinedOutputs>> {
    if !supported(execution, chunks, decode)? {
        return Ok(None);
    }
    let counts = chunks
        .iter()
        .map(|row| row.tokens.len())
        .chain(decode.iter().map(|_| 1))
        .collect::<Vec<_>>();
    execution.prefill_batches.clear();
    execution.combined.prepare(&execution.template, &counts)?;
    for row in chunks {
        if row.offset == 0 {
            let session = execution.template.instantiate_with_caches(&execution.caches)?;
            execution.sessions.insert(row.request.session_id, session);
        }
        execution.arm_terminal_checkpoint(row.request, row.offset, row.tokens.len())?;
    }
    let ids = chunks
        .iter()
        .map(|row| row.request.session_id)
        .chain(decode.iter().map(|row| row.session_id));
    let mut owned = take_sessions(&mut execution.sessions, ids)?;
    let result: Result<CombinedOutputs> = (|| {
        let tokens = chunks
            .iter()
            .flat_map(|row| row.tokens.iter().copied())
            .chain(decode.iter().map(|row| row.token_id))
            .collect::<Vec<_>>();
        let tables = chunks
            .iter()
            .map(|row| row.table)
            .chain(decode.iter().map(|row| &row.block_table))
            .collect::<Vec<_>>();
        let starts = chunks
            .iter()
            .map(|row| row.offset)
            .chain(owned.iter().skip(chunks.len()).map(|(_, session)| session.position()))
            .collect::<Vec<_>>();
        let policies = chunks
            .iter()
            .map(|row| {
                if row.final_chunk {
                    row.request.sampling_logits
                } else {
                    SamplingLogits::None
                }
            })
            .chain(decode.iter().map(|row| row.sampling_logits))
            .collect::<Vec<_>>();
        let mut sessions = owned.iter_mut().map(|(_, session)| session).collect::<Vec<_>>();
        let batch = execution.combined.current()?;
        let selected = batch
            .execute(&mut sessions, &tokens, &tables, &starts, Some(&policies))?
            .ok_or(Error::InvalidExecutionPlan("mixed step did not sample outputs"))?;
        let output = |token| Output { token: Some(token), logits: None };
        Ok(CombinedOutputs {
            prefill: chunks
                .iter()
                .zip(&selected)
                .map(|(row, token)| row.final_chunk.then(|| output(*token)))
                .collect(),
            decode: selected.into_iter().skip(chunks.len()).map(output).collect(),
        })
    })();
    restore_sessions(&mut execution.sessions, owned);
    let result = result?;
    for row in chunks {
        execution.checkpoint_prefix(row.request)?;
    }
    Ok(Some(result))
}

fn supported(
    execution: &MixedMixerExecution,
    chunks: &[PrefillChunk<'_>],
    decode: &[DecodeSequence],
) -> Result<bool> {
    if !execution.template.supports_combined_generation()
        || chunks
            .iter()
            .map(|row| row.request.session_id)
            .chain(decode.iter().map(|row| row.session_id))
            .any(|id| {
                execution.sessions.get(&id).is_some_and(|session| session.position_delta() != 0)
            })
        || chunks.is_empty()
        || chunks.iter().any(|row| !device_sampling(row.request.sampling_logits))
        || decode.iter().any(|row| !device_sampling(row.sampling_logits))
    {
        return Ok(false);
    }
    let mut ids = chunks
        .iter()
        .map(|row| row.request.session_id)
        .chain(decode.iter().map(|row| row.session_id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::InvalidDecoderKernel("mixed step repeats a session"));
    }
    for row in chunks {
        if row.offset != 0 && !execution.sessions.contains_key(&row.request.session_id) {
            return Err(Error::InvalidDecoderKernel("mixed prefill session is missing"));
        }
    }
    if decode.iter().any(|row| !execution.sessions.contains_key(&row.session_id)) {
        return Err(Error::InvalidDecoderKernel("mixed decode session is missing"));
    }
    Ok(true)
}
